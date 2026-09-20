import { TextSignAPI } from './text_sign_api.js';

export class TextSignUI {
    constructor() {
        this.api = new TextSignAPI();
        this.sessionId = null;
        this.isProcessing = false;
        this.currentSequence = [];
        
        // UI Elements
        this.inputElem = document.getElementById('textSignInput');
        this.btnProcess = document.getElementById('btnTextSignProcess');
        this.btnReset = document.getElementById('btnTextSignReset');
        this.sequenceContainer = document.getElementById('textSignSequence');
        this.unsupportedContainer = document.getElementById('textSignUnsupported');
        this.unsupportedList = document.getElementById('textSignUnsupportedList');
        this.statusText = document.getElementById('textSignStatus');
    }

    async init() {
        if (!this.inputElem || !this.btnProcess) {
            console.error("Text-to-Sign UI elements not found.");
            return;
        }

        // Initialize session
        try {
            const session = await this.api.createSession();
            this.sessionId = session.session_id;
        } catch (e) {
            this._showError(`Failed to initialize session: ${e.message}`);
        }

        this.btnProcess.addEventListener('click', () => this.handleProcess());
        this.btnReset.addEventListener('click', () => this.handleReset());
        
        // Also allow enter key (without shift) to submit
        this.inputElem.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleProcess();
            }
        });
    }

    async handleProcess() {
        if (this.isProcessing) return;
        
        const text = this.inputElem.value.trim();
        if (!text) {
            this._showError("Please enter some text to convert.");
            return;
        }

        this.isProcessing = true;
        this.btnProcess.disabled = true;
        this.inputElem.disabled = true;
        this._updateStatus('Processing...', 'info');

        // Clear previous results immediately
        this.currentSequence = [];
        this.sequenceContainer.innerHTML = '';
        this.unsupportedContainer.classList.add('d-none');
        this.unsupportedList.innerHTML = '';

        try {
            if (!this.sessionId) {
                const session = await this.api.createSession();
                this.sessionId = session.session_id;
            } else {
                try {
                    const sessionState = await this.api.getSession(this.sessionId);
                    if (sessionState.state === 'COMPLETED' || sessionState.state === 'ERROR') {
                        await this.api.resetSession(this.sessionId);
                    } else if (sessionState.state === 'CLOSED') {
                        const session = await this.api.createSession();
                        this.sessionId = session.session_id;
                    }
                } catch (e) {
                    const session = await this.api.createSession();
                    this.sessionId = session.session_id;
                }
            }

            const response = await this.api.processText(this.sessionId, text);
            this.currentSequence = response.sequence || [];
            
            this._renderSequence(this.currentSequence);
            this._renderUnsupported(response.unsupported_tokens || []);
            
            this._updateStatus('Completed', 'success');
        } catch (e) {
            this._showError(e.message);
        } finally {
            this.isProcessing = false;
            this.btnProcess.disabled = false;
            this.inputElem.disabled = false;
        }
    }

    async handleReset() {
        if (this.isProcessing || !this.sessionId) return;
        
        try {
            await this.api.resetSession(this.sessionId);
            
            // Clear UI
            this.inputElem.value = '';
            this.currentSequence = [];
            this.sequenceContainer.innerHTML = '';
            this.unsupportedContainer.classList.add('d-none');
            this.unsupportedList.innerHTML = '';
            this._updateStatus('', '');
            
        } catch (e) {
            this._showError(`Failed to reset session: ${e.message}`);
        }
    }

    _renderSequence(sequence) {
        this.sequenceContainer.innerHTML = '';
        
        if (sequence.length === 0) {
            this.sequenceContainer.innerHTML = '<span class="text-muted">No valid signs found.</span>';
            return;
        }
        
        sequence.forEach((item, index) => {
            // Create a conceptual rendering of the SignSequenceItem
            const signBlock = document.createElement('div');
            signBlock.className = 'd-inline-flex flex-column align-items-center me-3 mb-2 p-2 glass-card bg-opacity-25';
            signBlock.style.minWidth = '80px';
            signBlock.dataset.signId = item.sign_id;
            signBlock.dataset.index = item.original_index;
            
            const signIdElem = document.createElement('strong');
            signIdElem.className = 'text-info';
            signIdElem.textContent = item.sign_id;
            
            const sourceTextElem = document.createElement('span');
            sourceTextElem.className = 'small text-muted';
            sourceTextElem.textContent = item.source_text;
            
            signBlock.appendChild(signIdElem);
            signBlock.appendChild(sourceTextElem);
            
            this.sequenceContainer.appendChild(signBlock);
            
            // Add an arrow if not the last item
            if (index < sequence.length - 1) {
                const arrowBlock = document.createElement('div');
                arrowBlock.className = 'd-inline-flex align-items-center me-3 mb-2 text-secondary';
                arrowBlock.innerHTML = '<i class="fa-solid fa-arrow-right"></i>';
                this.sequenceContainer.appendChild(arrowBlock);
            }
        });
    }

    _renderUnsupported(unsupportedTokens) {
        if (unsupportedTokens.length > 0) {
            this.unsupportedList.innerHTML = '';
            unsupportedTokens.forEach(token => {
                const tokenBadge = document.createElement('span');
                tokenBadge.className = 'badge bg-warning text-dark me-2 mb-2';
                tokenBadge.textContent = token;
                this.unsupportedList.appendChild(tokenBadge);
            });
            this.unsupportedContainer.classList.remove('d-none');
        } else {
            this.unsupportedContainer.classList.add('d-none');
            this.unsupportedList.innerHTML = '';
        }
    }

    _showError(message) {
        this._updateStatus(message, 'danger');
        
        // Also show a global toast/notification if the global function exists (from app.js)
        // Since we are decoupled, we just rely on our status element, but we can try 
        // to find the toast system if needed. We'll stick to local status.
    }

    _updateStatus(message, type) {
        if (!this.statusText) return;
        
        this.statusText.textContent = message;
        this.statusText.className = 'small mt-2';
        if (type === 'danger') this.statusText.classList.add('text-danger');
        else if (type === 'success') this.statusText.classList.add('text-success');
        else if (type === 'info') this.statusText.classList.add('text-info');
        else this.statusText.classList.add('text-muted');
    }
}
