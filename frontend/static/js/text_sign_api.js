/**
 * TextSignAPI
 * Handles all backend HTTP communication for the Text-to-Sign pipeline.
 * Isolated from DOM manipulation.
 */
export class TextSignAPI {
    constructor(baseUrl = '/api/text-to-sign') {
        this.baseUrl = baseUrl;
    }

    async createSession() {
        const response = await fetch(`${this.baseUrl}/sessions`, {
            method: 'POST'
        });
        return this._handleResponse(response);
    }

    async getSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}`);
        return this._handleResponse(response);
    }

    async processText(sessionId, text) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        return this._handleResponse(response);
    }

    async resetSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/reset`, {
            method: 'POST'
        });
        return this._handleResponse(response);
    }

    async closeSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/close`, {
            method: 'POST'
        });
        return this._handleResponse(response);
    }

    async _handleResponse(response) {
        if (!response.ok) {
            let errorDetail = 'Unknown error occurred.';
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorDetail;
            } catch (e) {
                // If not JSON, just keep default error
            }
            throw new Error(errorDetail);
        }
        return await response.json();
    }
}
