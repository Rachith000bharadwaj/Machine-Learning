let recognition = null;

document.addEventListener('DOMContentLoaded', () => {
    setupVoiceInput();
    checkModelStatus();

    document.getElementById('analyzeBtn').addEventListener('click', submitDiagnosis);
    document.getElementById('voiceBtn').addEventListener('click', startVoice);
});

function setupVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const voiceButton = document.getElementById('voiceBtn');

    if (!SpeechRecognition) {
        voiceButton.disabled = true;
        voiceButton.textContent = 'Voice not supported';
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        document.getElementById('symptoms').value = event.results[0][0].transcript;
        updateVoiceStatus('Voice input captured.', 'success');
    };

    recognition.onerror = (event) => {
        updateVoiceStatus(`Voice error: ${event.error}`, 'error');
    };

    recognition.onend = () => {
        voiceButton.textContent = 'Use Voice Input';
    };
}

function checkModelStatus() {
    fetch('/api/health')
        .then((response) => response.json())
        .then((data) => {
            const status = document.getElementById('modelStatus');
            status.textContent = data.ready ? data.status : `Prototype mode: ${data.status}`;
            status.classList.toggle('is-ready', data.ready);
        })
        .catch(() => {
            document.getElementById('modelStatus').textContent = 'Model status unavailable';
        });
}

function startVoice() {
    if (!recognition) return;

    document.getElementById('voiceBtn').textContent = 'Listening...';
    updateVoiceStatus('Listening...', 'info');
    recognition.start();
}

function updateVoiceStatus(message, type) {
    const status = document.getElementById('voiceStatus');
    status.textContent = message;
    status.className = type;
    setTimeout(() => {
        status.textContent = '';
        status.className = '';
    }, 3000);
}

function fillExample(text) {
    document.getElementById('symptoms').value = text;
}

function submitDiagnosis() {
    const symptoms = document.getElementById('symptoms').value.trim();
    const analyzeButton = document.getElementById('analyzeBtn');

    if (!symptoms) {
        alert('Please enter your symptoms.');
        return;
    }

    analyzeButton.disabled = true;
    document.getElementById('inputSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';

    fetch('/diagnose', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({symptoms}),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) {
                alert(data.error);
                reset();
                return;
            }
            showResults(data);
        })
        .catch(() => {
            alert('The analysis could not be completed. Please try again.');
            reset();
        })
        .finally(() => {
            analyzeButton.disabled = false;
        });
}

function showResults(data) {
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'block';

    document.getElementById('diagnosisName').textContent = data.primary_diagnosis;

    const confidence = document.getElementById('confidenceScore');
    confidence.textContent = `${data.confidence}%`;
    confidence.className = data.confidence >= 80 ? 'good' : data.confidence >= 60 ? 'warn' : 'risk';

    document.getElementById('urgencyBadge').textContent = data.urgency_level.toUpperCase();
    document.getElementById('evidenceCount').textContent = `${data.evidence_count} symptoms`;

    renderSymptoms(data.matched_symptoms || []);
    document.getElementById('actionText').textContent = data.suggested_action;
    renderPredictions(data.top_predictions || []);

    window.scrollTo({top: 0, behavior: 'smooth'});
}

function renderSymptoms(symptoms) {
    const symptomsList = document.getElementById('symptomsList');
    symptomsList.innerHTML = '';

    if (!symptoms.length) {
        const empty = document.createElement('p');
        empty.className = 'empty-state';
        empty.textContent = 'No exact symptom match found.';
        symptomsList.appendChild(empty);
        return;
    }

    symptoms.forEach((symptom) => {
        const tag = document.createElement('div');
        tag.className = 'symptom-tag';

        const icon = document.createElement('span');
        icon.className = 'check-mark';
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = '+';
        tag.appendChild(icon);
        tag.appendChild(document.createTextNode(` ${symptom}`));
        symptomsList.appendChild(tag);
    });
}

function renderPredictions(predictions) {
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';

    if (!predictions.length) {
        const empty = document.createElement('p');
        empty.className = 'empty-state';
        empty.textContent = 'Add more recognizable symptoms to generate ranked predictions.';
        predictionsList.appendChild(empty);
        return;
    }

    predictions.forEach((prediction, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';

        const label = document.createElement('span');
        label.textContent = `${index + 1}. ${prediction.disease}`;

        const score = document.createElement('strong');
        score.textContent = `${prediction.confidence}%`;

        item.appendChild(label);
        item.appendChild(score);
        predictionsList.appendChild(item);
    });
}

function newAnalysis() {
    reset();
    document.getElementById('symptoms').value = '';
}

function reset() {
    document.getElementById('inputSection').style.display = 'block';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    window.scrollTo({top: 0, behavior: 'smooth'});
}

function printReport() {
    window.print();
}
