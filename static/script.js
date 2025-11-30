let currentPage = 1;
let allSongs = [];
const songsPerPage = 10;

const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const changeImageBtn = document.getElementById('changeImageBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const uploadSection = document.getElementById('uploadSection');
const error = document.getElementById('error');
const songsContainer = document.getElementById('songsContainer');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const prevBtnBottom = document.getElementById('prevBtnBottom');
const nextBtnBottom = document.getElementById('nextBtnBottom');
const pageInfo = document.getElementById('pageInfo');
const pageInfoBottom = document.getElementById('pageInfoBottom');
const newSearchBtn = document.getElementById('newSearchBtn');
const analysisInfo = document.getElementById('analysisInfo');

// Upload area click
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// File input change
imageInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    }
});

// Handle file selection
function handleFileSelect(file) {
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Change image button
changeImageBtn.addEventListener('click', () => {
    imageInput.value = '';
    previewContainer.style.display = 'none';
    uploadBtn.disabled = true;
});

// Upload button
uploadBtn.addEventListener('click', async () => {
    const file = imageInput.files[0];
    if (!file) return;

    // Hide previous results and errors
    resultsSection.style.display = 'none';
    error.style.display = 'none';
    uploadSection.style.display = 'none';
    loading.style.display = 'block';

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get recommendations');
        }

        if (data.success) {
            allSongs = data.playlist || [];
            currentPage = 1;
            displayResults(data.emotion, data.scene, data.objects);
        } else {
            throw new Error('Failed to get recommendations');
        }
    } catch (err) {
        showError(err.message);
        uploadSection.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
});

// Display results
function displayResults(emotion, scene, objects) {
    // Show analysis info
    let emotionText = emotion ? emotion.charAt(0).toUpperCase() + emotion.slice(1) : 'None detected';
    let sceneText = scene ? scene.charAt(0).toUpperCase() + scene.slice(1) : 'Unknown';
    let objectText = 'None detected';
    if (objects && Array.isArray(objects) && objects.length > 0) {
        objectText = objects.map(obj => 
            obj.charAt(0).toUpperCase() + obj.slice(1)
        ).join(', ');
    } else if (objects && typeof objects === 'string') {
        objectText = objects; 
    }
    analysisInfo.innerHTML = `
        <h3>🎭 Image Analysis</h3>
        <p><strong>Face Emotion:</strong> ${emotionText}</p>
        <p><strong>Scene Detected:</strong> ${sceneText}</p>
        <p><strong>Objects Detected:</strong> ${objectText}</p>
        
    `;

    // Display songs for current page
    displayPage(currentPage);
    
    // Show results section
    resultsSection.style.display = 'block';
    uploadSection.style.display = 'none';
}

// Display page
function displayPage(page) {
    const startIndex = (page - 1) * songsPerPage;
    const endIndex = startIndex + songsPerPage;
    const pageSongs = allSongs.slice(startIndex, endIndex);

    songsContainer.innerHTML = '';

    pageSongs.forEach((song, index) => {
        const songNumber = startIndex + index + 1;
        const songCard = createSongCard(song, songNumber);
        songsContainer.appendChild(songCard);
    });

    // Update pagination
    const totalPages = Math.ceil(allSongs.length / songsPerPage);
    pageInfo.textContent = `Page ${page} of ${totalPages}`;
    pageInfoBottom.textContent = `Page ${page} of ${totalPages}`;

    // Update button states
    prevBtn.disabled = page === 1;
    nextBtn.disabled = page === totalPages;
    prevBtnBottom.disabled = page === 1;
    nextBtnBottom.disabled = page === totalPages;
}

// Create song card
function createSongCard(song, number) {
    const card = document.createElement('div');
    card.className = 'song-card';
    card.innerHTML = `
        <div class="song-number">${number}</div>
        <div class="song-title">${escapeHtml(song.title)}</div>
        <div class="song-artist">${escapeHtml(song.artist)}</div>
        <a href="${song.link}" target="_blank" class="song-link">🎵 Play on Spotify</a>
    `;
    return card;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Pagination handlers
prevBtn.addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage--;
        displayPage(currentPage);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});

nextBtn.addEventListener('click', () => {
    const totalPages = Math.ceil(allSongs.length / songsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        displayPage(currentPage);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});

prevBtnBottom.addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage--;
        displayPage(currentPage);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});

nextBtnBottom.addEventListener('click', () => {
    const totalPages = Math.ceil(allSongs.length / songsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        displayPage(currentPage);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});

// New search button
newSearchBtn.addEventListener('click', () => {
    resultsSection.style.display = 'none';
    uploadSection.style.display = 'block';
    imageInput.value = '';
    previewContainer.style.display = 'none';
    uploadBtn.disabled = true;
    allSongs = [];
    currentPage = 1;
    error.style.display = 'none';
});

// Show error
function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
    loading.style.display = 'none';
}

