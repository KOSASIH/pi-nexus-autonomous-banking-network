// financialLiteracy.js

class FinancialLiteracy {
    constructor() {
        this.resources = []; // Store financial literacy resources
    }

    // Add a new resource
    addResource(title, type, url, description) {
        const resource = {
            id: this.resources.length + 1,
            title,
            type,
            url,
            description,
        };
        this.resources.push(resource);
        console.log(`Resource added: ${title}`);
    }

    // Get all resources
    getAllResources() {
        return this.resources;
    }

    // Get resource by ID
    getResourceById(id) {
        return this.resources.find(resource => resource.id === id) || null;
    }

    // Search resources by title
    searchResourcesByTitle(searchTerm) {
        return this.resources.filter(resource => 
            resource.title.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }
}

// Example usage
const financialLiteracy = new FinancialLiteracy();
financialLiteracy.addResource('Understanding Credit Scores', 'Article', 'https://example.com/credit-scores', 'Learn about credit scores and how they affect your financial health.');
financialLiteracy.addResource('Investing 101', 'Video', 'https://example.com/investing-101', 'A beginner\'s guide to investing.');
financialLiteracy.addResource('Budgeting Basics', 'Course', 'https://example.com/budgeting-basics', 'Learn how to create and stick to a budget.');

console.log('All Resources:', financialLiteracy.getAllResources());
console.log('Search Results for "Investing":', financialLiteracy.searchResourcesByTitle('Investing'));
