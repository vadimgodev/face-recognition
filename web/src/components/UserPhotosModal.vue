<template>
  <div v-if="isOpen" class="modal-overlay" @click.self="closeModal">
    <div class="modal-container">
      <div class="modal-header">
        <h2>{{ userName }}'s Photos</h2>
        <button class="close-btn" @click="closeModal">&times;</button>
      </div>

      <div class="modal-body">
        <div v-if="loading" class="loading">
          <div class="spinner"></div>
          <p>Loading photos...</p>
        </div>

        <div v-else-if="error" class="alert alert-error">
          {{ error }}
        </div>

        <div v-else>
          <div class="stats-row">
            <div class="stat-item">
              <span class="stat-label">Total Photos:</span>
              <span class="stat-value">{{ photos.total_photos }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Enrolled:</span>
              <span class="stat-value">{{ photos.enrolled_count }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">Verified:</span>
              <span class="stat-value">{{ photos.verified_count }}</span>
            </div>
          </div>

          <div class="photos-grid">
            <div
              v-for="photo in sortedPhotos"
              :key="photo.id"
              class="photo-card"
              :class="{ 'enrolled-photo': photo.photo_type === 'enrolled' }"
            >
              <div class="photo-badge">
                {{ photo.photo_type === 'enrolled' ? '📌 Enrolled' : '✓ Verified' }}
              </div>
              <img
                :src="getImageUrl(photo.id)"
                :alt="`${userName} - ${photo.photo_type}`"
                @error="handleImageError"
              />
              <div class="photo-info">
                <div class="info-row">
                  <span class="label">Type:</span>
                  <span class="value">{{ photo.photo_type }}</span>
                </div>
                <div v-if="photo.verified_confidence" class="info-row">
                  <span class="label">Confidence:</span>
                  <span class="value confidence">{{ (photo.verified_confidence * 100).toFixed(1) }}%</span>
                </div>
                <div v-if="photo.verified_by_processor" class="info-row">
                  <span class="label">Processor:</span>
                  <span
                    class="value processor-label"
                    :class="{ 'processor-aws': photo.verified_by_processor.includes('+aws') }"
                    :title="getProcessorTooltip(photo.verified_by_processor)"
                  >
                    {{ photo.verified_by_processor.includes('+aws') ? '☁️' : '🖥️' }}
                    {{ formatProcessor(photo.verified_by_processor) }}
                  </span>
                </div>
                <div class="info-row">
                  <span class="label">Date:</span>
                  <span class="value">{{ formatDate(photo.photo_type === 'verified' ? photo.verified_at : photo.created_at) }}</span>
                </div>
                <button
                  class="delete-photo-btn"
                  @click="handleDeletePhoto(photo.id, photo.photo_type)"
                  :title="`Delete this ${photo.photo_type} photo`"
                >
                  🗑️ Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="modal-footer">
        <button class="btn btn-secondary" @click="closeModal">Close</button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { faceApi } from '../api/faceApi'

export default {
  name: 'UserPhotosModal',
  props: {
    isOpen: {
      type: Boolean,
      required: true
    },
    userName: {
      type: String,
      default: ''
    }
  },
  emits: ['close', 'photoDeleted'],
  setup(props, { emit }) {
    const photos = ref({
      photos: [],
      total_photos: 0,
      enrolled_count: 0,
      verified_count: 0
    })
    const loading = ref(false)
    const error = ref(null)

    const sortedPhotos = computed(() => {
      // Sort: enrolled first, then verified by date descending
      return [...photos.value.photos].sort((a, b) => {
        if (a.photo_type === 'enrolled' && b.photo_type !== 'enrolled') return -1
        if (a.photo_type !== 'enrolled' && b.photo_type === 'enrolled') return 1

        const dateA = new Date(a.photo_type === 'verified' ? a.verified_at : a.created_at)
        const dateB = new Date(b.photo_type === 'verified' ? b.verified_at : b.created_at)
        return dateB - dateA
      })
    })

    const loadPhotos = async () => {
      if (!props.userName) return

      loading.value = true
      error.value = null

      try {
        const response = await faceApi.getUserPhotos(props.userName)
        photos.value = response
      } catch (err) {
        error.value = err.response?.data?.detail || 'Failed to load photos'
      } finally {
        loading.value = false
      }
    }

    const getImageUrl = (photoId) => {
      return faceApi.getFaceImageUrl(photoId)
    }

    const handleImageError = (event) => {
      event.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23ddd" width="200" height="200"/%3E%3Ctext fill="%23999" x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E'
    }

    const formatDate = (dateString) => {
      if (!dateString) return 'N/A'
      const date = new Date(dateString)
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    }

    const handleDeletePhoto = async (photoId, photoType) => {
      if (!confirm(`Are you sure you want to delete this ${photoType} photo?`)) {
        return
      }

      try {
        await faceApi.deleteFace(photoId)
        await loadPhotos() // Reload photos
        emit('photoDeleted')
      } catch (err) {
        alert('Failed to delete photo: ' + (err.response?.data?.detail || err.message))
      }
    }

    const closeModal = () => {
      emit('close')
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
        return 'Verified using InsightFace + AWS Rekognition (medium confidence match)'
      } else if (processor === 'aws_rekognition') {
        return 'Verified using AWS Rekognition'
      } else {
        return 'Verified using InsightFace only (high confidence match, no AWS cost)'
      }
    }

    // Watch for modal open and load photos
    watch(() => props.isOpen, (newValue) => {
      if (newValue) {
        loadPhotos()
      }
    })

    // Load photos when userName changes
    watch(() => props.userName, () => {
      if (props.isOpen) {
        loadPhotos()
      }
    })

    return {
      photos,
      loading,
      error,
      sortedPhotos,
      getImageUrl,
      handleImageError,
      formatDate,
      handleDeletePhoto,
      closeModal,
      formatProcessor,
      getProcessorTooltip
    }
  }
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
}

.modal-container {
  background: white;
  border-radius: 15px;
  max-width: 1200px;
  width: 100%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 30px;
  border-bottom: 1px solid #eee;
}

.modal-header h2 {
  margin: 0;
  color: #333;
}

.close-btn {
  background: none;
  border: none;
  font-size: 2em;
  color: #999;
  cursor: pointer;
  padding: 0;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: background 0.3s;
}

.close-btn:hover {
  background: #f5f5f5;
  color: #333;
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 30px;
}

.loading {
  text-align: center;
  padding: 40px;
  color: #666;
}

.spinner {
  border: 3px solid #f3f3f3;
  border-top: 3px solid #667eea;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 20px auto;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.alert {
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.alert-error {
  background: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}

.stats-row {
  display: flex;
  gap: 20px;
  margin-bottom: 30px;
  flex-wrap: wrap;
}

.stat-item {
  flex: 1;
  min-width: 150px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 15px 20px;
  border-radius: 10px;
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.stat-label {
  font-size: 0.9em;
  opacity: 0.9;
}

.stat-value {
  font-size: 1.5em;
  font-weight: bold;
}

.photos-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}

.photo-card {
  background: #f8f9fa;
  border-radius: 10px;
  overflow: hidden;
  transition: transform 0.3s, box-shadow 0.3s;
  position: relative;
}

.photo-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
}

.enrolled-photo {
  border: 3px solid #667eea;
}

.photo-badge {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 5px 12px;
  border-radius: 20px;
  font-size: 0.85em;
  font-weight: 600;
  z-index: 1;
}

.photo-card img {
  width: 100%;
  height: 250px;
  object-fit: cover;
}

.photo-info {
  padding: 15px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid #eee;
}

.info-row:last-of-type {
  border-bottom: none;
}

.info-row .label {
  color: #666;
  font-size: 0.9em;
  font-weight: 500;
}

.info-row .value {
  color: #333;
  font-size: 0.9em;
  font-weight: 600;
}

.value.confidence {
  color: #28a745;
}

.value.processor-label {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: #e8eaf6;
  color: #5e35b1;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.85em;
  border: 1px solid #d1d9ff;
}

.value.processor-label.processor-aws {
  background: #fff3cd;
  color: #856404;
  border: 1px solid #ffeaa7;
}

.delete-photo-btn {
  width: 100%;
  margin-top: 10px;
  padding: 8px;
  background: #dc3545;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.9em;
  transition: background 0.3s;
}

.delete-photo-btn:hover {
  background: #c82333;
}

.modal-footer {
  padding: 20px 30px;
  border-top: 1px solid #eee;
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.btn {
  padding: 10px 25px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1em;
  transition: background 0.3s;
}

.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn-secondary:hover {
  background: #5a6268;
}
</style>
