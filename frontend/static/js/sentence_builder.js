/**
 * Module 8.6: Sentence Builder
 * 
 * Consumes NEW confirmed signs and builds an ordered sentence.
 */

export class SentenceBuilder {
    constructor() {
        this.words = [];
    }

    /**
     * Adds a word to the sentence.
     * @param {string} word - The confirmed sign text.
     */
    addWord(word) {
        if (word === null || word === undefined) return;
        
        const trimmedWord = String(word).trim();
        if (trimmedWord.length === 0) return;

        this.words.push(trimmedWord);
    }

    /**
     * Returns the array of words.
     * @returns {Array<string>}
     */
    getWords() {
        return [...this.words];
    }

    /**
     * Returns the complete sentence as a space-separated string.
     * @returns {string}
     */
    getSentence() {
        return this.words.join(" ");
    }

    /**
     * Clears the sentence state.
     */
    clear() {
        this.words = [];
    }
}
