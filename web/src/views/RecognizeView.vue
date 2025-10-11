<template>
  <div class="recognize-view">
    <div class="card">
      <h2>Recognize Face</h2>

      <div v-if="message.text" :class="'alert alert-' + message.type">
        {{ message.text }}
      </div>

      <form @submit.prevent="handleSubmit" class="recognize-form">
        <div class="form-group">
          <label>Upload Photo *</label>
          <div class="file-upload">
            <input
              type="file"
              id="recognizeFile"
              @change="handleFileSelect"
              accept="image/*"
              required
            />
            <label for="recognizeFile" class="file-upload-label">
              <span>📷 Click to select image or drag here</span>
            </label>
          </div>
          <img v-if="imagePreview" :src="imagePreview" class="preview-image" alt="Preview" />
        </div>

        <div class="form-group">
          <label for="threshold">Confidence Threshold</label>
          <input
            id="threshold"
            type="number"
            v-model.number="confidenceThreshold"
            min="0"
            max="1"
            step="0.1"
          />
          <small class="help-text">
            Minimum similarity score (0.6 recommended for InsightFace)
          </small>
        </div>

        <button type="submit" class="btn btn-primary" :disabled="loading">
          <span v-if="loading">Recognizing...</span>
          <span v-else">Recognize Face</span>
        </button>
      </form>

      <div v-if="results.length > 0" class="results">
        <h3>Recognition Results</h3>
        <div v-if="executionTime !== null" class="execution-time">
          ⏱️ Execution time: <strong>{{ executionTime }} sec</strong>
        </div>
        <div v-for="result in results" :key="result.face.id" class="result-item">
          <div class="result-info">
            <div class="result-name">{{ result.face.user_name }}</div>
            <div v-if="result.face.user_email" class="result-details">
              {{ result.face.user_email }}
            </div>
            <div v-if="result.processor" class="result-processor">
              <span
                class="processor-badge"
                :class="{ 'aws-badge': result.processor.includes('+aws') }"
                :title="getProcessorTooltip(result.processor)"
              >
                {{ result.processor.includes('+aws') ? '☁️' : '🖥️' }}
                {{ formatProcessor(result.processor) }}
              </span>
            </div>
          </div>
          <div class="result-metrics">
            <div class="result-confidence">
              {{ (result.similarity * 100).toFixed(1) }}%
            </div>
            <div v-if="livenessChecked" class="liveness-badge" :class="livenessClass" :title="livenessTooltip">
              {{ livenessIcon }} {{ livenessText }}
            </div>
            <div v-if="result.photo_captured" class="photo-captured-badge" title="Photo was automatically captured for verification">
              📸 Captured
            </div>
          </div>
        </div>
      </div>

      <div v-else-if="attempted && results.length === 0" class="alert alert-info">
        No matching faces found. Try lowering the confidence threshold or enroll this person first.
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { faceApi } from '../api/faceApi'

export default {
  name: 'RecognizeView',
  setup() {
    const imageFile = ref(null)
    const imagePreview = ref(null)
    const confidenceThreshold = ref(0.6)
    const loading = ref(false)
    const message = ref({ text: '', type: '' })
    const results = ref([])
    const attempted = ref(false)
    const executionTime = ref(null)
    const livenessResult = ref(null)

    const handleFileSelect = (event) => {
      const file = event.target.files[0]
      if (file) {
        imageFile.value = file
        const reader = new FileReader()
        reader.onload = (e) => {
          imagePreview.value = e.target.result
        }
        reader.readAsDataURL(file)
      }
    }

    const handleSubmit = async () => {
      if (!imageFile.value) {
        message.value = { text: 'Please select an image', type: 'error' }
        return
      }

      loading.value = true
      message.value = { text: '', type: '' }
      results.value = []
      attempted.value = false
      executionTime.value = null

      const formData = new FormData()
      formData.append('image', imageFile.value)
      formData.append('confidence_threshold', confidenceThreshold.value)
      formData.append('max_results', 10)

      try {
        const response = await faceApi.recognizeFace(formData)
        results.value = response.matches || []
        executionTime.value = response.execution_time || null
        livenessResult.value = response.liveness_result || null
        attempted.value = true

        if (results.value.length > 0) {
          message.value = {
            text: `Found ${results.value.length} match(es)!`,
            type: 'success'
          }
        }
      } catch (error) {
        message.value = {
          text: error.response?.data?.detail || 'Failed to recognize face',
          type: 'error'
        }
        attempted.value = true
      } finally {
        loading.value = false
      }
    }

    const formatProcessor = (processor) => {
      if (!processor) return 'Unknown'

      // Format processor name for display
      if (processor === 'aws_rekognition') {
        return 'AWS Rekognition'
      }

      // Handle InsightFace models
      if (processor.includes('+aws')) {
        return processor.replace('+aws', ' + AWS')
      }

      return processor
    }

    const getProcessorTooltip = (processor) => {
      if (!processor) return ''

      if (processor.includes('+aws')) {
        return 'Verified using InsightFace + AWS Rekognition'
      } else if (processor === 'aws_rekognition') {
        return 'Verified using AWS Rekognition'
      } else {
        return 'Verified using InsightFace only (fast, no AWS cost)'
      }
    }

    const livenessChecked = computed(() => {
      return livenessResult.value !== null
    })

    const livenessClass = computed(() => {
      if (!livenessResult.value) return ''
      return livenessResult.value.is_real ? 'liveness-real' : 'liveness-fake'
    })

    const livenessIcon = computed(() => {
      if (!livenessResult.value) return ''
      return livenessResult.value.is_real ? '✅' : '🚫'
    })

    const livenessText = computed(() => {
      if (!livenessResult.value) return ''
      const confidence = (livenessResult.value.confidence * 100).toFixed(1)
      return livenessResult.value.is_real ? `Real (${confidence}%)` : `Fake (${confidence}%)`
    })

    const livenessTooltip = computed(() => {
      if (!livenessResult.value) return ''
      return livenessResult.value.is_real
        ? 'Liveness check passed - Real person detected'
        : `Spoofing detected - ${livenessResult.value.spoofing_type || 'Unknown type'}`
    })

    return {
      imagePreview,
      confidenceThreshold,
      loading,
      message,
      results,
      attempted,
      executionTime,
      handleFileSelect,
      handleSubmit,
      formatProcessor,
      getProcessorTooltip,
      livenessChecked,
      livenessClass,
      livenessIcon,
      livenessText,
      livenessTooltip
    }
  }
}
</script>

<style scoped>
.recognize-view {
  max-width: 800px;
  margin: 0 auto;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #333;
}

.form-group input[type="number"] {
  width: 100%;
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 8px;
  font-size: 1em;
  transition: border-color 0.3s;
}

.form-group input:focus {
  outline: none;
  border-color: #667eea;
}

.help-text {
  color: #666;
  margin-top: 5px;
  display: block;
  font-size: 0.9em;
}

.file-upload {
  position: relative;
}

.file-upload input[type="file"] {
  display: none;
}

.file-upload-label {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  border: 2px dashed #667eea;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
  background: #f8f9fa;
}

.file-upload-label:hover {
  background: #e8eaf6;
  border-color: #764ba2;
}

.preview-image {
  max-width: 100%;
  max-height: 300px;
  border-radius: 8px;
  margin-top: 15px;
  display: block;
  margin-left: auto;
  margin-right: auto;
}

.btn {
  padding: 12px 30px;
  border: none;
  border-radius: 8px;
  font-size: 1em;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
  width: 100%;
}

.btn-primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.alert {
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.alert-success {
  background: #d4edda;
  color: #155724;
  border: 1px solid #c3e6cb;
}

.alert-error {
  background: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}

.alert-info {
  background: #d1ecf1;
  color: #0c5460;
  border: 1px solid #bee5eb;
}

.results {
  margin-top: 30px;
}

.results h3 {
  margin-bottom: 15px;
  color: #333;
}

.execution-time {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  padding: 12px 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  font-size: 1em;
  text-align: center;
  box-shadow: 0 4px 6px rgba(240, 147, 251, 0.3);
}

.execution-time strong {
  font-weight: 700;
  font-size: 1.2em;
}

.result-item {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.result-info {
  flex: 1;
}

.result-name {
  font-weight: 600;
  font-size: 1.1em;
  color: #333;
}

.result-details {
  color: #666;
  font-size: 0.9em;
  margin-top: 3px;
}

.result-processor {
  margin-top: 8px;
}

.processor-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: #e8eaf6;
  color: #5e35b1;
  padding: 4px 12px;
  border-radius: 15px;
  font-size: 0.85em;
  font-weight: 600;
  border: 1px solid #d1d9ff;
}

.processor-badge.aws-badge {
  background: #fff3cd;
  color: #856404;
  border: 1px solid #ffeaa7;
}

.result-metrics {
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-end;
}

.result-confidence {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 5px 15px;
  border-radius: 20px;
  font-weight: 600;
}

.photo-captured-badge {
  background: #d4edda;
  color: #155724;
  padding: 4px 10px;
  border-radius: 15px;
  font-size: 0.8em;
  font-weight: 600;
  border: 1px solid #c3e6cb;
}

.liveness-badge {
  padding: 6px 12px;
  border-radius: 15px;
  font-size: 0.85em;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 5px;
}

.liveness-badge.liveness-real {
  background: #d4edda;
  color: #155724;
  border: 1px solid #c3e6cb;
}

.liveness-badge.liveness-fake {
  background: #ffe6e6;
  color: #c00;
  border: 1px solid #ffcccc;
}
</style>
