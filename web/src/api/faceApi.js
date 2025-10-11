import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'
const API_TOKEN = import.meta.env.VITE_API_TOKEN || ''

const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,  // Send cookies and Basic Auth
  headers: {
    'Content-Type': 'application/json',
    'x-face-token': API_TOKEN  // API token authentication
  }
})

// Add response interceptor to handle auth errors
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      console.error('Authentication failed:', error.response.data)
    }
    return Promise.reject(error)
  }
)

export const faceApi = {
  // Enroll a new face
  async enrollFace(formData) {
    const response = await api.post('/faces/enroll', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'x-face-token': API_TOKEN
      }
    })
    return response.data
  },

  // Recognize a face
  async recognizeFace(formData) {
    const response = await api.post('/faces/recognize', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'x-face-token': API_TOKEN
      }
    })
    return response.data
  },

  // Get all faces
  async getFaces(limit = 100, offset = 0) {
    const response = await api.get('/faces', {
      params: { limit, offset }
    })
    return response.data
  },

  // Get face by ID
  async getFaceById(faceId) {
    const response = await api.get(`/faces/${faceId}`)
    return response.data
  },

  // Get face image URL
  getFaceImageUrl(faceId) {
    return `${API_BASE_URL}/faces/${faceId}/image`
  },

  // Delete a face
  async deleteFace(faceId) {
    const response = await api.delete(`/faces/${faceId}`)
    return response.data
  },

  // Get all photos for a user
  async getUserPhotos(userName) {
    const response = await api.get(`/faces/user/${encodeURIComponent(userName)}/photos`)
    return response.data
  },

  // Check API health
  async healthCheck() {
    const response = await axios.get('/health', {
      withCredentials: true,
      headers: {
        'x-face-token': API_TOKEN
      }
    })
    return response.data
  },

  // Webcam endpoints
  async startWebcam() {
    const response = await api.post('/webcam/start')
    return response.data
  },

  async stopWebcam() {
    const response = await api.post('/webcam/stop')
    return response.data
  },

  async getWebcamStatus() {
    const response = await api.get('/webcam/status')
    return response.data
  },

  getWebcamStreamUrl() {
    // EventSource doesn't support custom headers, so we need to pass token as query param
    // But for now, we'll rely on basic auth at Traefik level
    return `${API_BASE_URL}/webcam/stream`
  }
}
