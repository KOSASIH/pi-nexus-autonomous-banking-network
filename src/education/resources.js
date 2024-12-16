// education/resources.js
class Resource {
    constructor(title, link, description) {
        this.id = Resource.incrementId();
        this.title = title;
        this.link = link;
        this.description = description;
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

class Resources {
    constructor() {
        this.resources = []; // Store resources
    }

    addResource(title, link, description) {
        const resource = new Resource(title, link, description);
        this.resources.push(resource);
        console.log(`Resource added: ${title}`);
        return resource;
    }

    getResources() {
        return this.resources;
    }

    getResourceById(resourceId) {
        const resource = this.resources.find(r => r.id === resourceId);
        if (!resource) {
            throw new Error('Resource not found.');
        }
        return resource;
    }

    updateResource(resourceId, updatedDetails) {
        const resource = this.getResourceById(resourceId);
        Object.assign(resource, updatedDetails);
        console.log(`Resource ${resourceId} updated.`);
        return resource;
    }

    deleteResource(resourceId) {
        const index = this.resources.findIndex(r => r.id === resourceId);
        if (index === -1) {
            throw new Error('Resource not found.');
        }
        this.resources.splice(index, 1);
        console.log(`Resource ${resourceId} deleted.`);
    }
}

module.exports = Resources;
