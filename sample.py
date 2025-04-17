title = "Irony in Daily Life"

prompt = """
Step 1: Define Core Elements of Visual Irony
"First, identify 5-8 key variables (nodes) that are fundamental to detecting irony in images. For each variable, provide an ID and a clear description of what it represents in the context of visual irony."

Step 2: Establish Primary Causal Relationships
"Next, determine the direct causal relationships between the variables you identified. For each relationship, specify the source node, target node, and describe exactly how the source influences or affects the target in creating ironic meaning."

Step 3: Identify Irony Indicators
"Then, create a special node called 'PerceivedIrony' that represents the determination of whether irony is present. Establish relationships between your previously defined variables and this irony node, explaining specifically how each contributing element leads to the perception of irony."

Step 4: Define Contextual Influences
"Identify any contextual or cultural factors that might influence the interpretation of irony in images. Create nodes for these factors and establish their relationships with other elements in your graph."

Step 5: Format as Structured JSON
"Finally, organize your complete causal reasoning graph into JSON format with this structure:
Follow this template to construct the causal graph:
{{
    \"nodes\": [
        {{\"id\": \"node1\", \"name\": \"Variable Name\", \"description\": \"Explanation of this variable\"}}
    ],
    \"edges\": [
        {{\"source\": \"node1\", \"target\": \"node2\", \"type\": \"causal\", \"strength\": \"strong/moderate/weak\", \"explanation\": \"Reasoning for this causal relationship\"}}
    ]
}}

Using the instruction given, answer the question:
Following image is a meme with the Title: {title}
Why is this meme funny?

Make sure that the final answer is followed after 'FinalAnswerWithoutCode:'
""".format(title=title)

print(prompt)