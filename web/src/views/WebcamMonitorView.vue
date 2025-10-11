<template>
  <div class="webcam-monitor">
    <div class="header">
      <h1>Webcam Monitor</h1>
      <p class="subtitle">Real-time face recognition monitoring</p>
    </div>

    <div class="mode-selector">
      <div class="mode-tabs">
        <button
          @click="selectMode('browser')"
          :class="['mode-tab', { active: mode === 'browser' }]"
          :disabled="isRunning"
        >
          Browser Camera
          <span class="mode-description">Development/Testing</span>
        </button>
        <button
          @click="selectMode('daemon')"
          :class="['mode-tab', { active: mode === 'daemon' }]"
          :disabled="isRunning"
        >
          Daemon Mode
          <span class="mode-description">Production (Headless)</span>
        </button>
      </div>
    </div>

    <div class="controls">
      <button
        @click="toggleWebcam"
        :class="['control-btn', isRunning ? 'stop' : 'start']"
        :disabled="loading"
      >
        {{ isRunning ? 'Stop Webcam' : 'Start Webcam' }}
      </button>
      <button
        v-if="mode === 'daemon'"
        @click="refreshStatus"
        class="control-btn refresh"
        :disabled="loading"
      >
        Refresh Status
      </button>
    </div>

    <div v-if="error" class="error-message">
      {{ error }}
    </div>

    <div class="status-panel">
      <div class="status-item">
        <span class="label">Status:</span>
        <span :class="['value', statusClass]">{{ status }}</span>
      </div>
      <div class="status-item" v-if="isRunning">
        <span class="label">🛡️ Liveness:</span>
        <span :class="['value', 'liveness-status', livenessEnabled ? 'enabled' : 'disabled']">
          {{ livenessEnabled ? 'ENABLED' : 'DISABLED' }}
        </span>
      </div>
      <div class="status-item" v-if="isRunning && mode === 'daemon'">
        <span class="label">Camera ID:</span>
        <span class="value">{{ cameraId }}</span>
      </div>
      <div class="status-item" v-if="isRunning">
        <span class="label">FPS:</span>
        <span class="value">{{ fps }}</span>
      </div>
      <div class="status-item" v-if="inCooldown">
        <span class="label">Cooldown:</span>
        <span class="value cooldown">{{ cooldownRemaining }}s remaining</span>
      </div>
      <div class="status-item" v-if="lastRecognizedUser">
        <span class="label">Last User:</span>
        <span class="value">{{ lastRecognizedUser }}</span>
      </div>
    </div>

    <div class="video-section" v-if="isRunning">
      <div class="video-container">
        <!-- Browser mode: HTML5 video element -->
        <video
          v-if="mode === 'browser'"
          ref="videoRef"
          autoplay
          playsinline
          class="video-feed"
        ></video>

        <!-- Daemon mode: SSE stream images -->
        <img
          v-if="mode === 'daemon' && currentFrame"
          :src="currentFrame"
          alt="Webcam feed"
          class="video-feed"
        />

        <div v-if="!currentFrame && mode === 'daemon'" class="video-placeholder">
          <p>Connecting to webcam stream...</p>
        </div>

        <div v-if="inCooldown" class="cooldown-overlay">
          <div class="cooldown-message">
            <h3>Access Granted!</h3>
            <p>Resuming scan in {{ cooldownRemaining }}s...</p>
          </div>
        </div>

        <!-- Hidden canvas for browser mode frame capture -->
        <canvas ref="canvasRef" style="display: none;"></canvas>
      </div>
    </div>

    <div class="events-section">
      <h2>Recent Events</h2>
      <div class="events-list">
        <div class="events-header">
          <span class="header-time">Time</span>
          <span class="header-user">User</span>
          <span class="header-liveness">Liveness</span>
          <span class="header-confidence">Confidence</span>
          <span class="header-processor">Model</span>
          <span class="header-timing">Timing</span>
          <span class="header-action">Action</span>
        </div>
        <div
          v-for="(event, index) in recentEvents"
          :key="index"
          :class="['event-item', event.result]"
        >
          <span class="event-time">{{ formatTime(event.timestamp) }}</span>
          <span class="event-user">{{ getEventUserName(event) }}</span>
          <span class="event-liveness">{{ getLivenessIcon(event) }}</span>
          <span class="event-confidence">{{ (event.confidence * 100).toFixed(1) }}%</span>
          <span class="event-processor" :class="{ 'aws-used': event.processor && event.processor.includes('aws') }">
            {{ event.processor || '-' }}
          </span>
          <span class="event-timing">
            <span class="timing-detail" v-if="event.detection_time">Det: {{ (event.detection_time * 1000).toFixed(0) }}ms</span>
            <span class="timing-detail" v-if="event.recognition_time">Rec: {{ (event.recognition_time * 1000).toFixed(0) }}ms</span>
            <span class="timing-total" v-if="event.execution_time">{{ (event.execution_time * 1000).toFixed(0) }}ms</span>
          </span>
          <span class="event-action">{{ event.door_action }}</span>
        </div>
        <div v-if="recentEvents.length === 0" class="no-events">
          No events yet
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { useUserMedia } from '@vueuse/core'
import { faceApi } from '../api/faceApi'

// Mode selection
const mode = ref('browser')

// Refs for video/canvas elements
const videoRef = ref(null)
const canvasRef = ref(null)

// Browser mode - useUserMedia
const enabled = ref(false)
const { stream } = useUserMedia({
  enabled,
  constraints: {
    video: { facingMode: 'user' },
    audio: false
  }
})

// Watch stream and connect to video element
watch(stream, async (newStream) => {
  if (newStream && videoRef.value) {
    await nextTick()
    videoRef.value.srcObject = newStream
  }
})

// State
const isRunning = ref(false)
const loading = ref(false)
const error = ref(null)
const status = ref('stopped')
const cameraId = ref(0)
const fps = ref(2)
const cooldownSeconds = ref(5)
const inCooldown = ref(false)
const cooldownRemaining = ref(0)
const lastRecognizedUser = ref(null)
const currentFrame = ref(null)
const recentEvents = ref([])
const livenessEnabled = ref(true) // Liveness detection status

// Daemon mode state
let eventSource = null
let statusInterval = null
let captureInterval = null
let cooldownInterval = null
let isProcessing = ref(false) // Prevent concurrent API requests

// Computed
const statusClass = computed(() => isRunning.value ? 'running' : 'stopped')

// Methods
const selectMode = (newMode) => {
  if (isRunning.value) {
    error.value = 'Stop webcam before switching modes'
    return
  }
  mode.value = newMode
  error.value = null
}

const toggleWebcam = async () => {
  if (isRunning.value) {
    await stopWebcam()
  } else {
    await startWebcam()
  }
}

const startWebcam = async () => {
  loading.value = true
  error.value = null

  try {
    if (mode.value === 'browser') {
      await startBrowserMode()
    } else {
      await startDaemonMode()
    }
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || 'Failed to start webcam'
  } finally {
    loading.value = false
  }
}

const stopWebcam = async () => {
  loading.value = true
  error.value = null

  try {
    if (mode.value === 'browser') {
      await stopBrowserMode()
    } else {
      await stopDaemonMode()
    }
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || 'Failed to stop webcam'
  } finally {
    loading.value = false
  }
}

// ========== Browser Mode Methods ==========
const startBrowserMode = async () => {
  // Enable camera
  enabled.value = true
  isRunning.value = true
  status.value = 'running'

  // Wait for stream
  await nextTick()
  await new Promise(resolve => setTimeout(resolve, 1000))

  // Start frame capture at 2 FPS
  captureInterval = setInterval(() => {
    captureBrowserFrame()
  }, 500) // 2 FPS
}

const stopBrowserMode = async () => {
  isRunning.value = false
  status.value = 'stopped'

  if (captureInterval) {
    clearInterval(captureInterval)
    captureInterval = null
  }

  if (cooldownInterval) {
    clearInterval(cooldownInterval)
    cooldownInterval = null
  }

  enabled.value = false
}

const captureBrowserFrame = async () => {
  // Skip if in cooldown
  if (inCooldown.value) {
    return
  }

  // Skip if already processing a request
  if (isProcessing.value) {
    return
  }

  const video = videoRef.value
  const canvas = canvasRef.value

  if (!video || !canvas) {
    return
  }

  if (video.videoWidth === 0 || video.videoHeight === 0) {
    return
  }

  try {
    // Set canvas dimensions
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw frame
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (!blob) {
        console.error('Failed to create blob')
        return
      }

      try {
        isProcessing.value = true
        const result = await recognizeFace(blob)
        if (result) {
          processRecognitionResult(result)
        }
      } catch (err) {
        // Check if this is a liveness check failure
        if (err.response?.data?.detail && err.response.data.detail.includes('Liveness check failed')) {
          // Extract liveness error details
          const errorDetail = err.response.data.detail
          const isSpoofing = errorDetail.includes('appears to be fake')

          addEvent({
            timestamp: new Date().toISOString(),
            result: isSpoofing ? 'spoofing_detected' : 'liveness_detection_failed',
            user_name: isSpoofing ? '🚫 SPOOFING BLOCKED' : '⚠️ DETECTION FAILED',
            confidence: 0.0,
            door_action: 'denied',
            processor: '-',
            execution_time: 0,
            detection_time: 0,
            recognition_time: 0,
            notes: errorDetail
          })
        } else {
          console.error('Recognition error:', err)
        }
      } finally {
        isProcessing.value = false
      }
    }, 'image/jpeg', 0.85)
  } catch (err) {
    console.error('Capture error:', err)
  }
}

const recognizeFace = async (imageBlob) => {
  const formData = new FormData()
  formData.append('image', imageBlob, 'frame.jpg')
  formData.append('max_results', '1')
  formData.append('confidence_threshold', '0.6')

  // Don't catch errors here - let them bubble up for liveness handling
  const response = await faceApi.recognizeFace(formData)
  return response
}

const processRecognitionResult = (result) => {
  if (!result.success) {
    return
  }

  const matches = result.matches || []
  if (matches.length === 0) {
    return
  }

  const bestMatch = matches[0]
  const face = bestMatch.face
  const similarity = bestMatch.similarity

  addEvent({
    timestamp: new Date().toISOString(),
    result: 'success',
    user_name: face.user_name,
    confidence: similarity,
    door_action: similarity >= 0.8 ? 'unlocked' : 'denied',
    processor: bestMatch.processor || result.processor || 'unknown',  // Prioritize match processor (shows AWS usage)
    execution_time: result.execution_time || 0,
    detection_time: result.detection_time || 0,
    recognition_time: result.recognition_time || 0,
  })

  if (similarity >= 0.8) {
    lastRecognizedUser.value = face.user_name
    startCooldown()
  }
}

const startCooldown = () => {
  inCooldown.value = true
  cooldownRemaining.value = cooldownSeconds.value

  if (cooldownInterval) {
    clearInterval(cooldownInterval)
  }

  cooldownInterval = setInterval(() => {
    cooldownRemaining.value--
    if (cooldownRemaining.value <= 0) {
      clearInterval(cooldownInterval)
      cooldownInterval = null
      inCooldown.value = false
      cooldownRemaining.value = 0
    }
  }, 1000)
}

// ========== Daemon Mode Methods ==========
const startDaemonMode = async () => {
  const response = await faceApi.startWebcam()

  if (response.success) {
    isRunning.value = true
    status.value = 'running'
    cameraId.value = response.camera_id
    fps.value = response.fps
    cooldownSeconds.value = response.cooldown_seconds

    startStream()

    statusInterval = setInterval(() => {
      refreshStatus()
    }, 1000)
  }
}

const stopDaemonMode = async () => {
  const response = await faceApi.stopWebcam()

  if (response.success) {
    isRunning.value = false
    status.value = 'stopped'
    stopStream()

    if (statusInterval) {
      clearInterval(statusInterval)
      statusInterval = null
    }
  }
}

const refreshStatus = async () => {
  try {
    const response = await faceApi.getWebcamStatus()

    if (response.success) {
      isRunning.value = response.status === 'running'
      status.value = response.status
      cameraId.value = response.camera_id
      fps.value = response.fps
      cooldownSeconds.value = response.cooldown_seconds

      if (isRunning.value) {
        inCooldown.value = response.in_cooldown || false
        cooldownRemaining.value = Math.ceil(response.cooldown_remaining || 0)
        lastRecognizedUser.value = response.last_recognized_user || null
      }
    }
  } catch (err) {
    console.error('Failed to refresh status:', err)
  }
}

const startStream = () => {
  stopStream()

  const streamUrl = faceApi.getWebcamStreamUrl()
  eventSource = new EventSource(streamUrl, { withCredentials: true })

  eventSource.addEventListener('frame', (event) => {
    try {
      const data = JSON.parse(event.data)
      currentFrame.value = `data:image/jpeg;base64,${data.frame}`
      inCooldown.value = data.in_cooldown || false
      cooldownRemaining.value = Math.ceil(data.cooldown_remaining || 0)

      if (data.last_recognized_user) {
        lastRecognizedUser.value = data.last_recognized_user
      }
    } catch (err) {
      console.error('Error parsing frame data:', err)
    }
  })

  eventSource.addEventListener('recognition', (event) => {
    try {
      const data = JSON.parse(event.data)
      addEvent(data)
    } catch (err) {
      console.error('Error parsing recognition data:', err)
    }
  })

  eventSource.onerror = (err) => {
    console.error('EventSource error:', err)
    stopStream()
  }
}

const stopStream = () => {
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
  currentFrame.value = null
}

const addEvent = (event) => {
  recentEvents.value.unshift(event)
  if (recentEvents.value.length > 20) {
    recentEvents.value.pop()
  }
}

const formatTime = (timestamp) => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString()
}

const getEventUserName = (event) => {
  if (event.result === 'spoofing_detected') {
    return '🚫 SPOOFING BLOCKED'
  }
  if (event.result === 'liveness_detection_failed') {
    return '⚠️ DETECTION FAILED'
  }
  return event.user_name || 'Unknown'
}

const getLivenessIcon = (event) => {
  if (event.result === 'spoofing_detected') {
    return '🚫'
  }
  if (event.result === 'liveness_detection_failed') {
    return '⚠️'
  }
  if (event.result === 'success') {
    return '✅'
  }
  return '-'
}

// Lifecycle
onMounted(() => {
  if (mode.value === 'daemon') {
    refreshStatus()
  }
})

onBeforeUnmount(() => {
  stopStream()

  if (statusInterval) {
    clearInterval(statusInterval)
  }

  if (captureInterval) {
    clearInterval(captureInterval)
  }

  if (cooldownInterval) {
    clearInterval(cooldownInterval)
  }

  enabled.value = false
})
</script>

<style scoped>
.webcam-monitor {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  text-align: center;
  margin-bottom: 20px;
}

.header h1 {
  color: #2c3e50;
  margin-bottom: 5px;
}

.subtitle {
  color: #7f8c8d;
  font-size: 14px;
}

.mode-selector {
  margin-bottom: 20px;
}

.mode-tabs {
  display: flex;
  gap: 10px;
  justify-content: center;
}

.mode-tab {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 15px 30px;
  border: 2px solid #bdc3c7;
  border-radius: 8px;
  background: white;
  cursor: pointer;
  transition: all 0.3s;
  font-size: 16px;
  font-weight: 500;
  color: #2c3e50;
}

.mode-tab:hover:not(:disabled) {
  border-color: #3498db;
  background: #ecf0f1;
}

.mode-tab.active {
  border-color: #3498db;
  background: #3498db;
  color: white;
}

.mode-tab:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.mode-description {
  font-size: 12px;
  font-weight: 400;
  margin-top: 5px;
  opacity: 0.8;
}

.controls {
  display: flex;
  gap: 10px;
  justify-content: center;
  margin-bottom: 20px;
}

.control-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.control-btn.start {
  background: #27ae60;
  color: white;
}

.control-btn.start:hover {
  background: #229954;
}

.control-btn.stop {
  background: #e74c3c;
  color: white;
}

.control-btn.stop:hover {
  background: #c0392b;
}

.control-btn.refresh {
  background: #3498db;
  color: white;
}

.control-btn.refresh:hover {
  background: #2980b9;
}

.control-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.error-message {
  background: #e74c3c;
  color: white;
  padding: 12px;
  border-radius: 6px;
  margin-bottom: 20px;
  text-align: center;
}

.status-panel {
  background: #ecf0f1;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  justify-content: center;
}

.status-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 120px;
}

.status-item .label {
  font-size: 12px;
  color: #7f8c8d;
  text-transform: uppercase;
  margin-bottom: 5px;
}

.status-item .value {
  font-size: 18px;
  font-weight: 600;
  color: #2c3e50;
}

.status-item .value.running {
  color: #27ae60;
}

.status-item .value.stopped {
  color: #e74c3c;
}

.status-item .value.cooldown {
  color: #f39c12;
}

.status-item .value.liveness-status.enabled {
  color: #27ae60;
}

.status-item .value.liveness-status.disabled {
  color: #e74c3c;
}

.video-section {
  margin-bottom: 30px;
}

.video-container {
  position: relative;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  aspect-ratio: 4/3;
  max-height: 480px;
  margin: 0 auto;
}

.video-feed {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.video-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #ecf0f1;
}

.cooldown-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(46, 204, 113, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
}

.cooldown-message {
  text-align: center;
  color: white;
}

.cooldown-message h3 {
  font-size: 32px;
  margin-bottom: 10px;
}

.cooldown-message p {
  font-size: 20px;
}

.events-section {
  background: #fff;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.events-section h2 {
  margin-top: 0;
  margin-bottom: 15px;
  color: #2c3e50;
}

.events-list {
  max-height: 300px;
  overflow-y: auto;
}

.events-header {
  display: grid;
  grid-template-columns: 90px 1fr 60px 80px 140px 170px 90px;
  gap: 8px;
  padding: 10px;
  font-weight: 600;
  background: #ecf0f1;
  border-bottom: 2px solid #bdc3c7;
  font-size: 13px;
  color: #2c3e50;
}

.event-item {
  display: grid;
  grid-template-columns: 90px 1fr 60px 80px 140px 170px 90px;
  gap: 8px;
  padding: 10px;
  border-bottom: 1px solid #ecf0f1;
  align-items: center;
}

.event-item.success {
  background: #d5f4e6;
}

.event-item.failure {
  background: #fadbd8;
}

.event-item.spoofing_detected {
  background: #ffe6e6;
  border-left: 4px solid #c00;
}

.event-item.liveness_detection_failed {
  background: #fff3cd;
  border-left: 4px solid #856404;
}

.event-time {
  font-size: 12px;
  color: #7f8c8d;
}

.event-user {
  font-weight: 500;
  color: #2c3e50;
}

.event-liveness {
  text-align: center;
  font-size: 18px;
}

.event-confidence {
  text-align: right;
  font-size: 14px;
  color: #3498db;
}

.event-processor {
  font-size: 12px;
  color: #7f8c8d;
  font-family: monospace;
}

.event-processor.aws-used {
  color: #e67e22;
  font-weight: 600;
}

.event-processor.aws-used::after {
  content: " 💰";
  font-size: 10px;
}

.event-timing {
  display: flex;
  flex-direction: column;
  font-size: 11px;
  gap: 2px;
}

.timing-detail {
  color: #95a5a6;
}

.timing-total {
  color: #2c3e50;
  font-weight: 600;
}

.event-action {
  text-align: right;
  font-size: 12px;
  text-transform: uppercase;
  font-weight: 600;
}

.event-item.success .event-action {
  color: #27ae60;
}

.event-item.failure .event-action {
  color: #e74c3c;
}

.no-events {
  text-align: center;
  padding: 40px;
  color: #95a5a6;
}
</style>
