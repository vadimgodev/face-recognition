<template>
  <div class="enroll-view">
    <div class="card">
      <h2>Enroll New Face</h2>

      <div v-if="message.text" :class="'alert alert-' + message.type">
        {{ message.text }}
      </div>

      <form @submit.prevent="handleSubmit" class="enroll-form">
        <div class="form-group">
          <label for="userName">Full Name *</label>
          <input
            id="userName"
            type="text"
            v-model="form.userName"
            placeholder="Enter full name"
            required
          />
        </div>

        <div class="form-group">
          <label for="userEmail">Email (Optional)</label>
          <input
            id="userEmail"
            type="email"
            v-model="form.userEmail"
            placeholder="Enter email address"
          />
        </div>

        <div class="form-group">
          <label>Upload Photo *</label>
          <div class="file-upload">
            <input
              type="file"
              id="imageFile"
              @change="handleFileSelect"
              accept="image/*"
              required
            />
            <label for="imageFile" class="file-upload-label">
              <span>📷 Click to select image or drag here</span>
            </label>
          </div>
          <img v-if="imagePreview" :src="imagePreview" class="preview-image" alt="Preview" />
        </div>

        <button type="submit" class="btn btn-primary" :disabled="loading">
          <span v-if="loading">Enrolling...</span>
          <span v-else>Enroll Face</span>
        </button>
      </form>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { faceApi } from '../api/faceApi'

export default {
  name: 'EnrollView',
  setup() {
    const form = ref({
      userName: '',
      userEmail: ''
    })
    const imageFile = ref(null)
    const imagePreview = ref(null)
    const loading = ref(false)
    const message = ref({ text: '', type: '' })

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

      const formData = new FormData()
      formData.append('image', imageFile.value)
      formData.append('user_name', form.value.userName)
      if (form.value.userEmail) {
        formData.append('user_email', form.value.userEmail)
      }

      try {
        await faceApi.enrollFace(formData)
        message.value = {
          text: `Successfully enrolled ${form.value.userName}!`,
          type: 'success'
        }

        // Reset form
        form.value = { userName: '', userEmail: '' }
        imageFile.value = null
        imagePreview.value = null
        document.getElementById('imageFile').value = ''
      } catch (error) {
        message.value = {
          text: error.response?.data?.detail || 'Failed to enroll face',
          type: 'error'
        }
      } finally {
        loading.value = false
      }
    }

    return {
      form,
      imagePreview,
      loading,
      message,
      handleFileSelect,
      handleSubmit
    }
  }
}
</script>

<style scoped>
.enroll-view {
  max-width: 600px;
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

.form-group input[type="text"],
.form-group input[type="email"] {
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
</style>
