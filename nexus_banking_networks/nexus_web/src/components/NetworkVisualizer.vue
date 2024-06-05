<template>
  <div>
    <h1>Network Visualizer</h1>
    <svg width="800" height="600">
      <g v-for="node in nodes" :key="node.node_id">
        <circle :cx="node.x" :cy="node.y" r="10" fill="blue" />
        <text :x="node.x" :y="node.y" text-anchor="middle">
          {{ node.node_id }}
        </text>
      </g>
      <g v-for="edge in edges" :key="edge.edge_id">
        <line
          :x1="edge.node1.x"
          :y1="edge.node1.y"
          :x2="edge.node2.x"
          :y2="edge.node2.y"
          stroke="black"
          stroke-width="2"
        />
      </g>
    </svg>
  </div>
</template>

<script>
export default {
  data() {
    return {
      nodes: [],
      edges: [],
    };
  },
  mounted() {
    this.fetchData();
  },
  methods: {
    fetchData() {
      // Fetch data from NexusAPI
      axios
        .get("/banks")
        .then((response) => {
          this.nodes = response.data;
          this.fetchEdges();
        })
        .catch((error) => {
          console.error(error);
        });
    },
    fetchEdges() {
      // Fetch edges from NexusAPI
      axios
        .get("/banks/edges")
        .then((response) => {
          this.edges = response.data;
        })
        .catch((error) => {
          console.error(error);
        });
    },
  },
};
</script>
