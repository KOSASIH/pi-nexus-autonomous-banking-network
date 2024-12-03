// education/tutorials.js
class Tutorial {
    constructor(title, content, author) {
        this.id = Tutorial.incrementId();
        this.title = title;
        this.content = content;
        this.author = author;
        this.createdAt = new Date();
    }

    static incrementId() {
        if (!this.currentId) {
            this.currentId = 1;
        } else {
            this.currentId++;
        }
        return this.currentId;
    }
}

class Tutorials {
    constructor() {
        this.tutorials = []; // Store tutorials
    }

    createTutorial(title, content, author) {
        const tutorial = new Tutorial(title, content, author);
        this.tutorials.push(tutorial);
        console.log(`Tutorial created: ${title}`);
        return tutorial;
    }

    getTutorials() {
        return this.tutorials;
    }

    getTutorialById(tutorialId) {
        const tutorial = this.tutorials.find(t => t.id === tutorialId);
        if (!tutorial) {
            throw new Error('Tutorial not found.');
        }
        return tutorial;
    }

    updateTutorial(tutorialId, updatedContent) {
        const tutorial = this.getTutorialById(tutorialId);
        tutorial.content = updatedContent;
        console.log(`Tutorial ${tutorialId} updated.`);
        return tutorial;
    }

    deleteTutorial(tutorialId) {
        const index = this.tutorials.findIndex(t => t.id === tutorialId);
        if (index === -1) {
            throw new Error('Tutorial not found.');
        }
        this.tutorials.splice(index, 1);
        console.log(`Tutorial ${tutorialId} deleted.`);
    }
}

module.exports = Tutorials;
