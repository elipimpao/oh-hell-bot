/**
 * Main application entry point.
 */

class App {
    constructor() {
        this.playMode = new PlayMode(this);
        this.advisorMode = new AdvisorMode(this);
        this.currentMode = null;
    }

    init() {
        this.showSetup();
    }

    showSetup() {
        const content = document.getElementById('main-content');
        content.innerHTML = '';
        content.className = 'setup-view';

        // Mode tabs
        const tabs = document.createElement('div');
        tabs.className = 'mode-tabs';

        const playTab = document.createElement('button');
        playTab.className = 'mode-tab active';
        playTab.textContent = 'Play Game';
        playTab.addEventListener('click', () => {
            playTab.classList.add('active');
            advisorTab.classList.remove('active');
            this._showPlaySetup(content);
        });
        tabs.appendChild(playTab);

        const advisorTab = document.createElement('button');
        advisorTab.className = 'mode-tab';
        advisorTab.textContent = 'Advisor Mode';
        advisorTab.addEventListener('click', () => {
            advisorTab.classList.add('active');
            playTab.classList.remove('active');
            this._showAdvisorSetup(content);
        });
        tabs.appendChild(advisorTab);

        content.appendChild(tabs);

        // Setup container
        const setupContainer = document.createElement('div');
        setupContainer.id = 'setup-container';
        content.appendChild(setupContainer);

        // Default to play mode setup
        this._showPlaySetup(content);
    }

    _showPlaySetup(content) {
        const container = document.getElementById('setup-container');
        this.playMode.renderSetup(container);
    }

    _showAdvisorSetup(content) {
        const container = document.getElementById('setup-container');
        this.advisorMode.renderSetup(container);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.init();
});
