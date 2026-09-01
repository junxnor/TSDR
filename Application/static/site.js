const page = document.body.dataset.page;
const nav = document.querySelector('[data-nav="' + page + '"]');
if (nav) nav.classList.add('active');

const menu = document.getElementById('menuButton');
if (menu) {
  menu.addEventListener('click', () => {
    const links = document.getElementById('navLinks');
    const open = links.classList.toggle('open');
    menu.setAttribute('aria-expanded', String(open));
  });
}

const $ = id => document.getElementById(id);

if (page === 'detect') {
  const imageInput = $('image');
  const resultImage = $('resultImage');
  const resultList = $('resultList');
  const empty = $('empty');
  const progress = $('uploadProgress');
  let selectedPreviewUrl = null;

  $('preset').addEventListener('change', event => {
    $('conf').value = event.target.value;
  });

  imageInput.addEventListener('change', event => {
    const file = event.target.files[0];
    if (!file) {
      resultImage.hidden = true;
      resultImage.removeAttribute('src');
      resultList.innerHTML = '';
      empty.hidden = false;
      return;
    }

    document.querySelector('.dropzone strong').textContent = file.name;
    document.querySelector('.dropzone small').textContent = (file.size / 1048576).toFixed(1) + ' MB selected';

    if (selectedPreviewUrl) URL.revokeObjectURL(selectedPreviewUrl);
    selectedPreviewUrl = URL.createObjectURL(file);
    resultImage.src = selectedPreviewUrl;
    resultImage.alt = 'Selected road image preview';
    resultImage.hidden = false;
    resultList.innerHTML = '';
    empty.hidden = true;
  });

  $('uploadForm').addEventListener('submit', async event => {
    event.preventDefault();
    const button = $('uploadButton');
    button.disabled = true;
    button.textContent = 'Analysing...';
    progress.hidden = false;
    empty.hidden = true;

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: new FormData(event.target)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Prediction failed');
      showResult(data);
    } catch (error) {
      resultImage.hidden = true;
      resultList.innerHTML = '';
      empty.hidden = false;
      empty.innerHTML = '<h2>Could not analyse image</h2><p>' + escapeHtml(error.message) + '</p>';
    } finally {
      progress.hidden = true;
      button.disabled = false;
      button.textContent = 'Analyse image →';
    }
  });

  resultImage.addEventListener('click', () => openImageModal(resultImage.src));
  resultImage.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      openImageModal(resultImage.src);
    }
  });
}

function showResult(data) {
  $('empty').hidden = true;
  const image = $('resultImage');
  image.hidden = false;
  image.src = data.image;
  image.alt = 'Annotated traffic sign result';

  const list = $('resultList');
  if (!data.detections.length) {
    list.innerHTML = '<article class="result-row"><div><strong>No accepted sign detected</strong><small>Try a clearer image or the balanced profile.</small></div></article>';
    return;
  }

  list.innerHTML = data.detections.map(row => {
    const status = row.low_confidence ? 'Review recommended' : 'Accepted';
    return '<article class="result-row">' +
      '<div><strong>' + escapeHtml(row.display_name) + '</strong>' +
      '<small>' + escapeHtml(row.class_name) + ' · ' + status + '</small></div>' +
      '<span class="result-metric"><small>Detection</small>' + (row.det_conf * 100).toFixed(1) + '%</span>' +
      '<span class="result-metric"><small>Recognition</small>' + (row.cls_conf * 100).toFixed(1) + '%</span>' +
      '</article>';
  }).join('');
}

function openImageModal(src, caption = '') {
  if (!src) return;
  const modal = $('imageModal');
  const modalImage = $('modalImage');
  modalImage.src = src;
  $('modalCaption').textContent = caption;
  $('modalCaption').hidden = !caption;
  modal.hidden = false;
  $('closeImageModal').focus();
}

function closeImageModal() {
  const modal = $('imageModal');
  if (!modal) return;
  modal.hidden = true;
  $('modalImage').removeAttribute('src');
  $('modalCaption').textContent = '';
}

const modal = $('imageModal');
if (modal) {
  $('closeImageModal').addEventListener('click', closeImageModal);
  modal.addEventListener('click', event => {
    if (event.target === modal) closeImageModal();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !modal.hidden) closeImageModal();
  });
}

if (page === 'live') {
  let mediaStream = null;
  let timer = null;
  let busy = false;
  let liveSessionId = null;
  const spokenAt = new Map();
  const speechCooldownMs = 10000;
  const trafficSignSpeech = {
    1: 'U-turn ahead. Follow the direction carefully.', 2: 'Keep right.', 3: 'Keep left.',
    4: 'Obstruction ahead. Pass either side safely.', 5: 'Motorcycle track. Motorcyclists use the designated track.',
    6: 'Stop sign. Stop completely.', 7: 'No left turn.', 8: 'No right turn.', 9: 'No U-turn.',
    10: 'No entry. Do not enter.', 11: 'Five-tonne limit. Do not proceed if over the limit.',
    12: 'Thirty-tonne limit. Do not proceed if over the limit.',
    13: 'Two-metre height limit. Do not proceed if too tall.',
    14: 'Three-metre height limit. Do not proceed if too tall.',
    15: 'Four-metre height limit. Do not proceed if too tall.',
    16: 'Five-metre height limit. Do not proceed if too tall.',
    17: 'Six-metre height limit. Do not proceed if too tall.',
    18: 'Speed limit twenty. Slow down.', 19: 'Speed limit thirty. Slow down.',
    20: 'Speed limit forty. Slow down.', 21: 'Speed limit fifty. Slow down.',
    22: 'Speed limit sixty. Slow down.', 23: 'Speed limit seventy. Adjust your speed.',
    24: 'Speed limit eighty. Do not exceed the limit.', 25: 'Speed limit ninety. Do not exceed the limit.',
    26: 'Speed limit one hundred and ten. Do not exceed the limit.',
    27: 'No heavy vehicles. Use another route.', 28: 'Heavy vehicles keep left.',
    29: 'No parking.', 30: 'No stopping.', 31: 'Give way. Slow down and yield.',
    32: 'Three point five metre width limit. Do not proceed if too wide.',
    33: 'No overtaking. Stay behind the vehicle ahead.',
    34: 'Roadworks ahead. Slow down and follow instructions.',
    35: 'Camera zone. Observe the speed limit.',
    36: 'Crosswind ahead. Slow down and hold the steering firmly.',
    37: 'Road hump. Slow down now.', 38: 'Road hump ahead. Slow down.', 39: 'Towing zone. Do not park.',
    40: 'Left bend ahead. Slow down.', 41: 'Slippery road. Slow down and avoid sudden braking.',
    42: 'Pedestrian crossing. Slow down and prepare to stop.',
    43: 'Pedestrian crossing. Slow down and prepare to stop.',
    44: 'Children crossing. Slow down and prepare to stop.',
    45: 'Children crossing. Slow down and prepare to stop.',
    46: 'Caution ahead. Slow down and watch the road.',
    47: 'Road narrows on the left. Slow down and keep your distance.',
    48: 'Traffic lights ahead. Slow down and prepare to stop.',
    49: 'Obstacle ahead. Slow down and be ready to steer safely.',
    50: 'Staggered junctions ahead. Slow down and watch for traffic.',
    51: 'T-junction ahead. Slow down and prepare to stop.',
    52: 'Road joins from the right. Slow down and watch for traffic.',
    53: 'Road joins from the left. Slow down and watch for traffic.',
    54: 'Left exit ahead. Move left safely if exiting.',
    55: 'Crossroads ahead. Slow down and check all directions.',
    56: 'Minor road on the right. Watch for entering vehicles.',
    57: 'Minor road on the left. Watch for entering vehicles.',
    58: 'Minor road on the left. Watch for entering vehicles.',
    59: 'Cattle crossing. Slow down and prepare to stop.',
    60: 'Roundabout ahead. Slow down and give way.',
    61: 'Narrow bridge. Slow down and keep your distance.',
    62: 'Road divides ahead. Use the correct lane.',
    63: 'Two-way traffic. Keep left and watch for oncoming vehicles.',
    64: 'Divided road ending. Prepare for two-way traffic.',
    65: 'Left curve ahead. Slow down.', 66: 'Y-junction ahead. Slow down and choose your direction.'
  };
  const urgentSigns = new Set([6, 10, 31, 34, 37, 38, 41, 42, 43, 44, 45, 48, 49, 51, 55, 59, 60, 61, 63]);

  function signNumber(row) {
    const match = String(row.class_name || '').match(/sign_(\d+)/i);
    return match ? Number(match[1]) : null;
  }

  function preferredVoice() {
    const voices = window.speechSynthesis?.getVoices() || [];
    return voices.find(voice => voice.lang.toLowerCase() === 'en-my') ||
      voices.find(voice => voice.lang.toLowerCase().startsWith('en-gb')) ||
      voices.find(voice => voice.lang.toLowerCase().startsWith('en')) || null;
  }

  function speakDetections(detections) {
    if (!$('voiceAlerts').checked || !('speechSynthesis' in window)) return;
    const now = Date.now();
    const alerts = detections
      .filter(row => row.recognition_accepted && !row.low_confidence)
      .map(row => ({ row, number: signNumber(row) }))
      .filter(item => item.number && trafficSignSpeech[item.number])
      .filter(item => {
        const row = item.row;
        const key = row.class_name || row.display_name;
        return now - (spokenAt.get(key) || 0) >= speechCooldownMs;
      })
      .sort((a, b) => Number(urgentSigns.has(b.number)) - Number(urgentSigns.has(a.number)))
      .slice(0, 3);
    if (!alerts.length) return;

    alerts.forEach(item => spokenAt.set(item.row.class_name || item.row.display_name, now));
    window.speechSynthesis.cancel();
    const message = alerts.map(item => trafficSignSpeech[item.number]).join(' ');
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = 'en-MY';
    utterance.rate = alerts.length > 1 ? 1.3 : 1.12;
    utterance.pitch = 1;
    const voice = preferredVoice();
    if (voice) utterance.voice = voice;
    window.speechSynthesis.speak(utterance);
  }

  $('startLive').addEventListener('click', event => {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      event.stopImmediatePropagation();
      $('cameraMessage').textContent = 'Phone camera blocked: browsers permit camera access only on HTTPS or localhost. Open the deployed HTTPS site instead of the 192.168.x.x HTTP address.';
    }
  }, true);

  $('startLive').addEventListener('click', async () => {
    try {
      liveSessionId = window.crypto?.randomUUID?.() ||
        Date.now().toString(36) + Math.random().toString(36).slice(2);
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false
      });
      const video = $('liveVideo');
      video.srcObject = mediaStream;
      await video.play();
      video.hidden = false;
      $('cameraPlaceholder').hidden = true;
      $('liveBadge').textContent = '● Live';
      $('liveBadge').classList.add('online');
      $('cameraMessage').textContent = 'Rear camera is running. Processing one frame about every 1.2 seconds.';
      if ($('voiceAlerts').checked && 'speechSynthesis' in window) {
        window.speechSynthesis.resume();
      }
      timer = setInterval(processDeviceFrame, 1200);
    } catch (error) {
      $('cameraMessage').textContent = 'Camera could not start: ' + error.message;
    }
  });

  $('stopLive').addEventListener('click', stopDeviceCamera);

  async function processDeviceFrame() {
    if (busy || !mediaStream) return;
    const video = $('liveVideo');
    if (!video.videoWidth) return;
    busy = true;
    const canvas = $('captureCanvas');
    const scale = Math.min(1, 960 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async blob => {
      try {
        const form = new FormData();
        form.append('image', blob, 'mobile-frame.jpg');
        form.append('conf', $('cameraConf').value);
        form.append('min_cls_conf', $('cameraMinClsConf').value);
        form.append('source_type', 'live');
        form.append('live_session_id', liveSessionId);
        const response = await fetch('/predict', { method: 'POST', body: form });
        const data = await response.json();
        if (response.ok && data.detections.length) {
          speakDetections(data.detections);
          const result = $('liveResult');
          result.src = data.image;
          result.hidden = false;
          video.hidden = true;
          setTimeout(() => {
            result.hidden = true;
            video.hidden = false;
          }, 850);
          $('liveDetectionList').innerHTML = data.detections.map(row =>
            '<article class="result-row"><div><strong>' + escapeHtml(row.display_name) +
            '</strong><small>' + escapeHtml(row.class_name) + '</small></div></article>'
          ).join('');
        }
      } finally {
        busy = false;
      }
    }, 'image/jpeg', .82);
  }

  function stopDeviceCamera() {
    if (timer) clearInterval(timer);
    timer = null;
    if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
    if (liveSessionId) {
      const form = new FormData();
      form.append('live_session_id', liveSessionId);
      fetch('/stop_live_session', { method: 'POST', body: form });
      liveSessionId = null;
    }
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    spokenAt.clear();
    $('liveVideo').hidden = true;
    $('liveResult').hidden = true;
    $('cameraPlaceholder').hidden = false;
    $('liveBadge').textContent = '● Offline';
    $('liveBadge').classList.remove('online');
    $('cameraMessage').textContent = 'Camera stopped.';
  }
}

let recordData = { upload: [], live: [] };
let recordType = 'upload';
if (page === 'records') {
  document.querySelectorAll('[data-record]').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('[data-record]').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    recordType = button.dataset.record;
    renderRecords();
  }));
  $('refreshRecords').addEventListener('click', loadRecords);
  loadRecords();
}

async function loadRecords() {
  try {
    const response = await fetch('/records');
    recordData = await response.json();
    renderRecords();
  } catch {
    $('recordCount').textContent = 'Records unavailable';
  }
}

function renderRecords() {
  const rows = recordData[recordType] || [];
  $('recordCount').textContent = rows.length + ' record' + (rows.length === 1 ? '' : 's');
  const empty = '<tr><td colspan="5">No detections recorded yet.</td></tr>';
  $('recordTableBody').innerHTML = rows.length ? rows.map(recordRow).join('') : empty;
  $('recordList').innerHTML = rows.length ? rows.map(mobileRecord).join('') : '<p>No detections recorded yet.</p>';
  document.querySelectorAll('[data-record-index]').forEach(item => {
    const openRecord = () => {
      const record = rows[Number(item.dataset.recordIndex)];
      if (record) {
        const signDetails = recordDetections(record).map(sign =>
          sign.display_name + ' (Detection ' + (Number(sign.det_conf) * 100).toFixed(1) +
          '%, Recognition ' + (Number(sign.cls_conf) * 100).toFixed(1) + '%)'
        ).join(' · ');
        const caption = formatTime(record.timestamp) + ' · ' + signDetails;
        openImageModal(record.image_url, caption);
      }
    };
    item.addEventListener('click', openRecord);
    item.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openRecord();
      }
    });
  });
  document.querySelectorAll('[data-delete-record]').forEach(button => {
    button.addEventListener('keydown', event => event.stopPropagation());
    button.addEventListener('click', async event => {
      event.stopPropagation();
      const record = rows[Number(button.dataset.deleteRecord)];
      if (!record || !window.confirm('Delete this record permanently?')) return;
      button.disabled = true;
      try {
        const response = await fetch('/records/' + encodeURIComponent(record.id), { method: 'DELETE' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Could not delete record');
        recordData[recordType] = rows.filter(row => row.id !== record.id);
        renderRecords();
      } catch (error) {
        button.disabled = false;
        window.alert(error.message);
      }
    });
  });
}

function recordRow(record, index) {
  return '<tr class="record-review" data-record-index="' + index + '" tabindex="0" role="button">' +
    '<td><img class="record-thumb" src="' + escapeHtml(record.image_url) + '" alt="Recorded frame"></td>' +
    '<td>' + formatTime(record.timestamp) + '</td>' +
    '<td>' + recordSignsHtml(record) + '</td>' +
    '<td><span class="review-link">View details →</span></td>' +
    '<td><button class="delete-record" type="button" data-delete-record="' + index + '">Delete</button></td></tr>';
}

function mobileRecord(record, index) {
  return '<article class="mobile-record record-review" data-record-index="' + index + '" tabindex="0" role="button">' +
    '<img src="' + escapeHtml(record.image_url) + '" alt="Recorded frame">' +
    '<div>' + recordSignsHtml(record) +
    '<small>' + formatTime(record.timestamp) + '<br>Tap to review</small>' +
    '<button class="delete-record" type="button" data-delete-record="' + index + '">Delete record</button></div></article>';
}

function recordDetections(record) {
  if (Array.isArray(record.detections)) return record.detections;
  return [record];
}

function recordSignsHtml(record) {
  return recordDetections(record).map(sign =>
    '<div class="record-sign"><strong>' + escapeHtml(sign.display_name) + '</strong>' +
    '<small>' + escapeHtml(sign.class_name) + '</small></div>'
  ).join('');
}

function formatTime(value) {
  const date = new Date(Number(value) * 1000);
  return Number.isNaN(date.valueOf()) ? escapeHtml(value) : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  }[char]));
}
