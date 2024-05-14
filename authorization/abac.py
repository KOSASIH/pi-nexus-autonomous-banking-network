from py_abac import AttributeProvider, Inquiry, Policy, RulesChecker


class AttributeProvider(AttributeProvider):
    def get_attribute_value(self, ace, attribute_path, ctx):
        if ace == "subject" and attribute_path == "role":
            return "admin"
        elif ace == "resource" and attribute_path == "category":
            return "finance"


policy = Policy(
    2,
    actions=["read", "write"],
    resources=[{"category": "finance"}],
    subjects=[{"role": "admin"}],
    effect=Policy.ALLOW_ACCESS,
    description="Allow admins to read and write financial data",
)

storage = MemoryStorage()
storage.add(policy)

guard = Guard(storage, AttributeProvider(), RulesChecker())

inquiry = Inquiry(
    action="read",
    resource={"id": "456", "category": "finance"},
    subject={"id": "123", "role": "admin"},
)

assert guard.is_allowed(inquiry)
