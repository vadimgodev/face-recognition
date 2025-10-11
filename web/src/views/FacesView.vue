<template>
  <div class="faces-view">
    <div class="card">
      <h2>Enrolled Users</h2>

      <div class="stats">
        <div class="stat-card">
          <div class="stat-value">{{ uniqueUsers }}</div>
          <div class="stat-label">Unique Users</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ totalPhotos }}</div>
          <div class="stat-label">Total Photos</div>
        </div>
      </div>

      <div v-if="loading" class="loading">
        <div class="spinner"></div>
        <p>Loading faces...</p>
      </div>

      <div v-else-if="enrolledUsers.length === 0" class="alert alert-info">
        No faces enrolled yet. Go to "Enroll Face" to add faces.
      </div>

      <div v-else class="faces-grid">
        <div v-for="user in enrolledUsers" :key="user.userName" class="face-card">
          <div class="photo-count-badge">{{ user.totalPhotos }} photo{{ user.totalPhotos > 1 ? 's' : '' }}</div>
          <img
            :src="getFaceImageUrl(user.enrolledPhoto.id)"
            :alt="user.userName"
            @error="handleImageError"
          />
          <h3>{{ user.userName }}</h3>
          <p v-if="user.enrolledPhoto.user_email" class="email">{{ user.enrolledPhoto.user_email }}</p>
          <div class="photo-details">
            <div class="detail-item">
              <span class="icon">📌</span>
              <span>{{ user.enrolledCount }} enrolled</span>
            </div>
            <div class="detail-item">
              <span class="icon">✓</span>
              <span>{{ user.verifiedCount }} verified</span>
            </div>
          </div>
          <div class="action-buttons">
            <button class="view-btn" @click="openUserPhotos(user.userName)">
              View All Photos
            </button>
            <button class="delete-btn" @click="handleDeleteUser(user.userName)">
              Delete User
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- User Photos Modal -->
    <UserPhotosModal
      :is-open="isModalOpen"
      :user-name="selectedUserName"
      @close="closeModal"
      @photo-deleted="handlePhotoDeleted"
    />
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { faceApi } from '../api/faceApi'
import UserPhotosModal from '../components/UserPhotosModal.vue'

export default {
  name: 'FacesView',
  components: {
    UserPhotosModal
  },
  setup() {
    const faces = ref([])
    const totalPhotos = ref(0)
    const loading = ref(false)
    const isModalOpen = ref(false)
    const selectedUserName = ref('')

    // Group faces by user and get enrolled photo for each
    const enrolledUsers = computed(() => {
      const userMap = new Map()

      faces.value.forEach(face => {
        if (!userMap.has(face.user_name)) {
          userMap.set(face.user_name, {
            userName: face.user_name,
            enrolledPhoto: null,
            enrolledCount: 0,
            verifiedCount: 0,
            totalPhotos: 0
          })
        }

        const user = userMap.get(face.user_name)
        user.totalPhotos++

        if (face.photo_type === 'enrolled') {
          user.enrolledPhoto = face
          user.enrolledCount++
        } else if (face.photo_type === 'verified') {
          user.verifiedCount++
        }
      })

      // Convert to array and filter out users without enrolled photo
      return Array.from(userMap.values())
        .filter(user => user.enrolledPhoto !== null)
        .sort((a, b) => a.userName.localeCompare(b.userName))
    })

    const uniqueUsers = computed(() => {
      return enrolledUsers.value.length
    })

    const loadFaces = async () => {
      loading.value = true
      try {
        const response = await faceApi.getFaces()
        faces.value = response.faces || []
        totalPhotos.value = response.total || 0
      } catch (error) {
        console.error('Failed to load faces:', error)
      } finally {
        loading.value = false
      }
    }

    const getFaceImageUrl = (faceId) => {
      return faceApi.getFaceImageUrl(faceId)
    }

    const handleImageError = (event) => {
      event.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23ddd" width="200" height="200"/%3E%3Ctext fill="%23999" x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E'
    }

    const openUserPhotos = (userName) => {
      selectedUserName.value = userName
      isModalOpen.value = true
    }

    const closeModal = () => {
      isModalOpen.value = false
      selectedUserName.value = ''
    }

    const handlePhotoDeleted = async () => {
      // Reload faces when a photo is deleted
      await loadFaces()
    }

    const handleDeleteUser = async (userName) => {
      if (!confirm(`Are you sure you want to delete all photos for ${userName}?`)) {
        return
      }

      try {
        // Get all photos for this user
        const userPhotos = faces.value.filter(f => f.user_name === userName)

        // Delete all photos
        for (const photo of userPhotos) {
          await faceApi.deleteFace(photo.id)
        }

        await loadFaces()
      } catch (error) {
        alert('Failed to delete user: ' + (error.response?.data?.detail || error.message))
      }
    }

    onMounted(() => {
      loadFaces()
    })

    return {
      faces,
      totalPhotos,
      uniqueUsers,
      enrolledUsers,
      loading,
      isModalOpen,
      selectedUserName,
      getFaceImageUrl,
      handleImageError,
      openUserPhotos,
      closeModal,
      handlePhotoDeleted,
      handleDeleteUser
    }
  }
}
</script>

<style scoped>
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
}

.stat-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  border-radius: 10px;
  text-align: center;
}

.stat-value {
  font-size: 2em;
  font-weight: bold;
  margin-bottom: 5px;
}

.stat-label {
  font-size: 0.9em;
  opacity: 0.9;
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

.alert-info {
  background: #d1ecf1;
  color: #0c5460;
  border: 1px solid #bee5eb;
}

.faces-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.face-card {
  background: #f8f9fa;
  border-radius: 10px;
  padding: 15px;
  text-align: center;
  transition: transform 0.3s, box-shadow 0.3s;
  position: relative;
}

.face-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.15);
}

.photo-count-badge {
  position: absolute;
  top: 20px;
  right: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.85em;
  font-weight: 600;
  z-index: 1;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.face-card img {
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 8px;
  margin-bottom: 10px;
}

.face-card h3 {
  color: #333;
  margin-bottom: 5px;
  font-size: 1.2em;
}

.face-card .email {
  color: #666;
  font-size: 0.85em;
  margin: 5px 0;
}

.photo-details {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin: 15px 0;
  padding: 10px;
  background: white;
  border-radius: 8px;
}

.detail-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.9em;
  color: #555;
}

.detail-item .icon {
  font-size: 1.1em;
}

.action-buttons {
  display: flex;
  gap: 10px;
  margin-top: 15px;
}

.view-btn {
  flex: 1;
  padding: 10px 15px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.9em;
  font-weight: 600;
  transition: transform 0.2s, box-shadow 0.2s;
}

.view-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.delete-btn {
  flex: 1;
  padding: 10px 15px;
  background: #dc3545;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 0.9em;
  font-weight: 600;
  transition: background 0.3s;
}

.delete-btn:hover {
  background: #c82333;
}
</style>
