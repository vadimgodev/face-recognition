import { createRouter, createWebHistory } from 'vue-router'
import EnrollView from '../views/EnrollView.vue'
import RecognizeView from '../views/RecognizeView.vue'
import FacesView from '../views/FacesView.vue'
import WebcamMonitorView from '../views/WebcamMonitorView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/enroll'
    },
    {
      path: '/enroll',
      name: 'enroll',
      component: EnrollView
    },
    {
      path: '/recognize',
      name: 'recognize',
      component: RecognizeView
    },
    {
      path: '/faces',
      name: 'faces',
      component: FacesView
    },
    {
      path: '/monitoring',
      name: 'monitoring',
      component: WebcamMonitorView
    }
  ]
})

export default router
