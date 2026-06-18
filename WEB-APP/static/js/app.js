let recognition = null;

const state = {
    lastResult: null,
};

document.addEventListener('DOMContentLoaded', () => {
    setupVoiceInput();
    setupInteractions();
    checkModelStatus();
});

function setupInteractions() {
    document.getElementById('analyzeBtn').addEventListener('click', submitDiagnosis);
    document.getElementById('voiceBtn').addEventListener('click', startVoice);
    document.getElementById('newAnalysisBtn').addEventListener('click', newAnalysis);
    document.getElementById('printBtn').addEventListener('click', printReport);

    document.querySelectorAll('[data-example]').forEach((button) => {
        button.addEventListener('click', () => {
            document.getElementById('symptoms').value = button.dataset.example || '';
            document.getElementById('symptoms').focus();
        });
    });
}

function setupVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const voiceButton = document.getElementById('voiceBtn');

    if (!SpeechRecognition) {
        voiceButton.disabled = true;
        voiceButton.innerHTML = `${iconMarkup('X')} Voice`;
        updateVoiceStatus('Voice unavailable in this browser. Typed offline intake is available.', 'error', false);
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        document.getElementById('symptoms').value = event.results[0][0].transcript;
        updateVoiceStatus('Voice captured.', 'success');
    };

    recognition.onerror = (event) => {
        updateVoiceStatus(`Voice error: ${event.error}`, 'error');
    };

    recognition.onend = () => {
        voiceButton.innerHTML = `${iconMarkup('V')} Voice`;
    };
}

function checkModelStatus() {
    fetch('/api/health')
        .then((response) => response.json())
        .then((data) => {
            const status = document.getElementById('modelStatus');
            const ehrMiniStatus = document.getElementById('ehrMiniStatus');
            const fractureMiniStatus = document.getElementById('fractureMiniStatus');
            const featureStatus = [
                data.ehr_ready ? 'EHR index ready' : 'EHR index unavailable',
                data.fracture_ready ? 'fracture screen ready' : 'fracture screen unavailable',
                data.secure_logging ? 'encrypted logs' : 'plain logs',
            ];
            const statusText = data.ready
                ? `${data.status}; ${featureStatus.join('; ')}`
                : `Prototype mode: ${data.status}`;

            status.innerHTML = `<span class="status-dot"></span><span>${statusText}</span>`;
            status.classList.toggle('ready', data.ready);
            ehrMiniStatus.textContent = data.ehr_ready ? 'on' : 'off';
            fractureMiniStatus.textContent = data.fracture_ready ? 'on' : 'off';
        })
        .catch(() => {
            document.getElementById('modelStatus').innerHTML = '<span class="status-dot"></span><span>Model status unavailable</span>';
            document.getElementById('ehrMiniStatus').textContent = 'off';
            document.getElementById('fractureMiniStatus').textContent = 'off';
        });
}

function startVoice() {
    if (!recognition) return;

    document.getElementById('voiceBtn').innerHTML = `${iconMarkup('S')} Listening`;
    updateVoiceStatus('Listening...', 'info', false);
    recognition.start();
}

function updateVoiceStatus(message, type, autoClear = true) {
    const status = document.getElementById('voiceStatus');
    status.textContent = message;
    status.className = `voice-status ${type}`;

    if (autoClear) {
        setTimeout(() => {
            status.textContent = '';
            status.className = 'voice-status';
        }, 3000);
    }
}

function submitDiagnosis() {
    const symptoms = document.getElementById('symptoms').value.trim();
    const analyzeButton = document.getElementById('analyzeBtn');

    if (!symptoms) {
        updateVoiceStatus('Enter symptoms first.', 'error');
        document.getElementById('symptoms').focus();
        return;
    }

    setLoading(true);
    analyzeButton.disabled = true;

    fetch('/diagnose', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({symptoms}),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) {
                throw new Error(data.error);
            }
            state.lastResult = data;
            showResults(data);
        })
        .catch((error) => {
            showEmptyState();
            updateVoiceStatus(error.message || 'Analysis failed.', 'error');
        })
        .finally(() => {
            setLoading(false);
            analyzeButton.disabled = false;
        });
}

function setLoading(isLoading) {
    if (isLoading) {
        document.getElementById('emptyState').hidden = true;
        document.getElementById('resultContent').hidden = true;
    }
    document.getElementById('loadingSection').hidden = !isLoading;
}

function showEmptyState() {
    document.getElementById('emptyState').hidden = false;
    document.getElementById('loadingSection').hidden = true;
    document.getElementById('resultContent').hidden = true;
}

function showResults(data) {
    document.getElementById('emptyState').hidden = true;
    document.getElementById('loadingSection').hidden = true;
    document.getElementById('resultContent').hidden = false;

    document.getElementById('diagnosisName').textContent = data.primary_diagnosis || '-';
    renderUrgency(data.urgency_level || 'Needs Review');
    renderMetrics(data);
    renderSymptoms(data.matched_symptoms || []);
    renderEhrEvidence(data.ehr_evidence || {});
    renderPredictions(data.top_predictions || []);

    document.getElementById('actionText').textContent = data.suggested_action || '-';
    document.getElementById('resultsSection').scrollIntoView({behavior: 'smooth', block: 'start'});
}

function renderUrgency(urgency) {
    const badge = document.getElementById('urgencyBadge');
    const className = urgency.toLowerCase().replace(/\s+/g, '-');
    badge.textContent = urgency;
    badge.className = `urgency-badge ${className}`;
}

function renderMetrics(data) {
    const confidence = Number(data.confidence || 0);
    const confidenceElement = document.getElementById('confidenceScore');
    const caseCount = data.ehr_evidence?.matched_cases || 0;

    confidenceElement.textContent = `${confidence.toFixed(1).replace('.0', '')}%`;
    confidenceElement.className = confidence >= 80 ? 'good' : confidence >= 60 ? 'warn' : 'risk';
    document.getElementById('evidenceCount').textContent = `${data.evidence_count || 0}`;
    document.getElementById('caseCount').textContent = formatNumber(caseCount);
}

function renderSymptoms(symptoms) {
    const list = document.getElementById('symptomsList');
    list.innerHTML = '';

    if (!symptoms.length) {
        list.appendChild(emptyMessage('No exact symptom match found.'));
        return;
    }

    symptoms.forEach((symptom) => {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.innerHTML = `${iconMarkup('OK')}<span>${escapeHtml(symptom)}</span>`;
        list.appendChild(tag);
    });
}

function renderEhrEvidence(evidence) {
    const box = document.getElementById('ehrEvidence');
    box.innerHTML = '';

    const lead = document.createElement('div');
    lead.className = 'evidence-lead';
    lead.innerHTML = `<strong>${formatNumber(evidence.matched_cases || 0)}</strong><span>matched reference cases</span>`;
    box.appendChild(lead);

    const sources = evidence.source_counts || {};
    const sourceNames = Object.keys(sources);
    if (sourceNames.length) {
        const sourceList = document.createElement('div');
        sourceList.className = 'source-list';
        sourceNames.forEach((sourceName) => {
            const source = document.createElement('span');
            source.className = 'source-pill';
            source.textContent = `${formatSourceName(sourceName)}: ${formatNumber(sources[sourceName])}`;
            sourceList.appendChild(source);
        });
        box.appendChild(sourceList);
    }

    const supportingSymptoms = evidence.supporting_symptoms || [];
    if (supportingSymptoms.length) {
        const supportGrid = document.createElement('div');
        supportGrid.className = 'support-grid';
        supportingSymptoms.forEach((item) => {
            const row = document.createElement('div');
            row.className = 'support-row';
            row.innerHTML = `<span>${escapeHtml(item.symptom)}</span><strong>${formatNumber(item.support_count || 0)}</strong>`;
            supportGrid.appendChild(row);
        });
        box.appendChild(supportGrid);
    }

    if (evidence.description) {
        const description = document.createElement('p');
        description.className = 'reference-text';
        description.textContent = evidence.description;
        box.appendChild(description);
    }
}

function renderPredictions(predictions) {
    const list = document.getElementById('predictionsList');
    list.innerHTML = '';

    if (!predictions.length) {
        list.appendChild(emptyMessage('No ranked predictions available.'));
        return;
    }

    predictions.forEach((prediction, index) => {
        const row = document.createElement('div');
        row.className = 'prediction-row';
        row.innerHTML = `<span>${index + 1}. ${escapeHtml(prediction.disease)}</span><strong>${formatConfidence(prediction.confidence)}</strong>`;
        list.appendChild(row);
    });
}

function newAnalysis() {
    state.lastResult = null;
    document.getElementById('symptoms').value = '';
    showEmptyState();
    document.getElementById('symptoms').focus();
}

function printReport() {
    window.print();
}

function emptyMessage(text) {
    const element = document.createElement('p');
    element.className = 'reference-text';
    element.textContent = text;
    return element;
}

function formatNumber(value) {
    return Number(value || 0).toLocaleString('en-US');
}

function formatConfidence(value) {
    const number = Number(value || 0);
    return `${number.toFixed(1).replace('.0', '')}%`;
}

function formatSourceName(name) {
    const names = {
        augmented_symptom_reference: 'Augmented',
        patient_profile: 'Profiles',
        syditriage: 'Triage',
        testing: 'Test',
        training: 'Train',
    };
    return names[name] || name.replace(/_/g, ' ');
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function iconMarkup(label) {
    return `<span class="app-icon" data-icon="${label}" aria-hidden="true"></span>`;
}
