from dataclasses import dataclass

@dataclass
class StellarModel:
    def to_xdr(self):
        raise NotImplementedError

    @classmethod
    def from_xdr(cls, xdr_data):
        raise NotImplementedError
