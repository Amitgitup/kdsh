CLAIM_VERIFICATION_PROMPT = """
You are a literary reasoning assistant evaluating narrative consistency.

Your task is NOT strict fact-checking.
Your task is to judge whether a claim is COMPATIBLE with the narrative evidence.

Definitions:
- CONSISTENT: The evidence supports or aligns with the claim, even if some details are implied rather than explicitly stated.
- CONTRADICT: The evidence clearly conflicts with the claim.
- UNCLEAR: The evidence does not provide enough information to reasonably judge the claim.

Important rules:
1. Do NOT require every detail of the claim to be explicitly stated.
2. If the narrative portrayal reasonably supports the claim and nothing contradicts it, choose CONSISTENT.
3. Absence of a minor detail does NOT make a claim unclear.
4. Only choose CONTRADICT if the evidence clearly disagrees with the claim.
5. Prefer CONSISTENT over UNCLEAR when evidence aligns overall.

Claim:
{claim}

Relevant excerpts from the novel:
{evidence_blocks}

First, write a brief analysis explaining how the evidence relates to the claim.

Then provide your final decision.

YOU MUST end your response with BOTH of the following lines.
Do not omit them.
Do not add anything after them.

Final Label: CONSISTENT or CONTRADICT or UNCLEAR
Final Explanation: One or two sentences explaining your decision.
"""



#############################################################################################
CLAIM_DECOMPOSITION_PROMPT = """
You are performing INFORMATION EXTRACTION, not summarization.

Your task is to extract ALL explicit factual statements from the claim.

Rules (IMPORTANT):
- Each sub-claim MUST express exactly ONE factual assertion.
- DO NOT merge multiple facts into one sentence.
- DO NOT summarize or shorten.
- DO NOT drop details.
- Preserve names, relations, actions, and attributes exactly as stated.
- If the claim contains N distinct facts, output N sub-claims.

Claim:
"{claim}"

Output format (STRICT):
Return ONLY a numbered list.
Each line MUST start with a number and a period.

Example format:
1. Fact one.
2. Fact two.
3. Fact three.
"""
