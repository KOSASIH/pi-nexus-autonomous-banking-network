import { ComplianceProtocol } from "../ComplianceProtocol";
import { RegulatoryKnowledgeGraph } from "../regulatory-knowledge-graph/RegulatoryKnowledgeGraph";
import { ReportingAnalytics } from "../reporting-analytics/ReportingAnalytics";
import { MockRegulatoryKnowledgeGraph } from "./MockRegulatoryKnowledgeGraph";
import { MockReportingAnalytics } from "./MockReportingAnalytics";

describe("ComplianceProtocol", () => {
  let complianceProtocol;
  let regulatoryKnowledgeGraph;
  let reportingAnalytics;

  beforeEach(async () => {
    regulatoryKnowledgeGraph = new MockRegulatoryKnowledgeGraph();
    reportingAnalytics = new MockReportingAnalytics();
    complianceProtocol = new ComplianceProtocol(regulatoryKnowledgeGraph, reportingAnalytics);
    await complianceProtocol.init();
  });

  it("should initialize correctly", async () => {
    expect(complianceProtocol.regulatoryKnowledgeGraph).toBe(regulatoryKnowledgeGraph);
    expect(complianceProtocol.reportingAnalytics).toBe(reportingAnalytics);
  });

  it("should evaluate compliance rules correctly", async () => {
    const rule = {
      id: "rule-1",
      description: "Test rule",
      regulatoryText: "Test regulatory text",
      applicableTo: ["entity-1", "entity-2"],
    };
    await regulatoryKnowledgeGraph.addComplianceRule(rule);

    const result = await complianceProtocol.evaluateComplianceRule(rule.id, "entity-1");
    expect(result.status).toBe("COMPLIANT");
  });

  it("should generate compliance report correctly", async () => {
    const reportConfig = {
      query: "SELECT * FROM compliance_rules WHERE applicableTo = 'entity-1'",
    };
    const report = await complianceProtocol.generateComplianceReport(reportConfig);
    expect(report.visualizations).toHaveLength(1);
    expect(report.summary).toBe("Compliance report for entity-1");
  });

  it("should stream compliance data correctly", async () => {
    const stream = await complianceProtocol.streamComplianceData("entity-1");
    expect(stream).toBeInstanceOf(AsyncGenerator);

    const data = await stream.next();
    expect(data.value).toEqual({
      ruleId: "rule-1",
      entityId: "entity-1",
      status: "COMPLIANT",
    });
  });
});

class MockRegulatoryKnowledgeGraph {
  async init() {}

  async addComplianceRule(rule) {
    // Mock implementation
  }

  async getComplianceRule(id) {
    // Mock implementation
  }
}

class MockReportingAnalytics {
  async init() {}

  async generateReport(reportConfig) {
    // Mock implementation
  }

  async streamRealtimeData(reportConfig) {
    // Mock implementation
  }
}
