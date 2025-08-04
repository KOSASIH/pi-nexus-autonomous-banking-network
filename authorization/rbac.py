from py_abac import AttributeProvider, Inquiry, Policy, RulesChecker


class RoleProvider(AttributeProvider):
    def get_attribute_value(self, ace, attribute_path, ctx):
        if ace == "subject" and attribute_path == "role":
            return "admin"


policy = Policy(
    1,
    actions=["read", "write"],
    resources=["data"],
    subjects=[{"role": "admin"}],
    effect=Policy.ALLOW_ACCESS,
    description="Allow admins to read and write data",
)

storage = MemoryStorage()
storage.add(policy)

guard = Guard(storage, RoleProvider(), RulesChecker())

inquiry = Inquiry(
    action="read", resource="data", subject={"id": "123", "role": "admin"}
)

assert guard.is_allowed(inquiry)
