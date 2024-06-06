module Cybersecurity where

import Data.List
import Data.Maybe
import qualified Data.ByteString as B

type Rule = (B.ByteString, B.ByteString)

cybersecurityRules :: [Rule]
cybersecurityRules = [
    ("malware", "alert"),
    ("virus", "alert"),
    ("trojan", "alert")
    ]

detectThreats :: B.ByteString -> [B.ByteString]
detectThreats data = catMaybes $ map (detectThreat data) cybersecurityRules

detectThreat :: B.ByteString -> Rule -> Maybe B.ByteString
detectThreat data (pattern, alert) =
    if B.isInfixOf pattern data then Just alert else Nothing

-- Example usage:
data = B.pack "This is a sample data with a malware"
threats = detectThreats data
print threats
