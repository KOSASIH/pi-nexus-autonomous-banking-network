class KnowledgeGraph {
  constructor() {
    this.entities = {};
    this.relations = {};
  }

  async addEntity(entity) {
    // Add entity to knowledge graph
    this.entities[entity.id] = entity;
  }

  async addRelation(relation) {
    // Add relation to knowledge graph
    this.relations[relation.id] = relation;
  }

  async getEntity(id) {
    // Get entity from knowledge graph
    return this.entities[id];
  }

  async getRelation(id) {
    // Get relation from knowledge graph
    return this.relations[id];
  }
}

export { KnowledgeGraph };
