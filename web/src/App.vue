<template>
  <div id="app">
    <header class="app-header">
      <div class="container">
        <h1>Face Recognition System</h1>
        <p>Powered by InsightFace + Smart Hybrid Recognition</p>
      </div>
    </header>

    <nav class="app-nav">
      <div class="container">
        <router-link to="/enroll" class="nav-link">
          <span class="nav-icon">👤</span>
          <span>Enroll Face</span>
        </router-link>
        <router-link to="/recognize" class="nav-link">
          <span class="nav-icon">🔍</span>
          <span>Recognize Face</span>
        </router-link>
        <router-link to="/faces" class="nav-link">
          <span class="nav-icon">📋</span>
          <span>View Faces</span>
        </router-link>
        <router-link to="/monitoring" class="nav-link">
          <span class="nav-icon">📹</span>
          <span>Webcam Monitor</span>
        </router-link>
      </div>
    </nav>

    <main class="app-main">
      <div class="container">
        <router-view />
      </div>
    </main>

    <footer class="app-footer">
      <div class="container">
        <p>&copy; 2025 Face Recognition System</p>
      </div>
    </footer>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

.app-header {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  color: white;
  padding: 30px 0;
  text-align: center;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.app-header h1 {
  font-size: 2.5em;
  margin-bottom: 10px;
}

.app-header p {
  font-size: 1.1em;
  opacity: 0.9;
}

.app-nav {
  background: rgba(255, 255, 255, 0.95);
  padding: 15px 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  position: sticky;
  top: 0;
  z-index: 100;
}

.app-nav .container {
  display: flex;
  gap: 15px;
  justify-content: center;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: white;
  border: 2px solid #e0e0e0;
  border-radius: 10px;
  text-decoration: none;
  color: #333;
  font-weight: 600;
  transition: all 0.3s;
}

.nav-link:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  border-color: #667eea;
}

.nav-link.router-link-active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: transparent;
}

.nav-icon {
  font-size: 1.3em;
}

.app-main {
  flex: 1;
  padding: 30px 0;
}

.app-footer {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  color: white;
  text-align: center;
  padding: 20px 0;
  margin-top: auto;
}

.card {
  background: white;
  border-radius: 15px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  margin-bottom: 20px;
}

.card h2 {
  color: #333;
  margin-bottom: 20px;
}

@media (max-width: 768px) {
  .app-header h1 {
    font-size: 1.8em;
  }

  .app-nav .container {
    flex-direction: column;
  }

  .nav-link {
    justify-content: center;
  }
}
</style>
