# Chapter 1: Prompt Chaining

*Original content: 12 pages - by Antonio Gulli*

## Prompt Chaining Pattern Overview

Prompt chaining, sometimes referred to as Pipeline pattern, represents a powerful paradigm for handling intricate tasks when leveraging large language models (LLMs). Rather than expecting an LLM to solve a complex problem in a single, monolithic step, prompt chaining advocates for a divide-and-conquer strategy. The core idea is to break down the original, daunting problem into a sequence of smaller, more manageable sub-problems. Each sub-problem is addressed individually through a specifically designed prompt, and the output generated from one prompt is strategically fed as input into the subsequent prompt in the chain.

This sequential processing technique inherently introduces modularity and clarity into the interaction with LLMs. By decomposing a complex task, it becomes easier to understand and debug each individual step, making the overall process more robust and interpretable. Each step in the chain can be meticulously crafted and optimized to focus on a specific aspect of the larger problem, leading to more accurate and focused outputs.

The output of one step acting as the input for the next is crucial. This passing of information establishes a dependency chain, hence the name, where the context and results of previous operations guide the subsequent processing. This allows the LLM to build on its previous work, refine its understanding, and progressively move closer to the desired solution.

Furthermore, prompt chaining is not just about breaking down problems; it also enables the integration of external knowledge and tools. At each step, the LLM can be instructed to interact with external systems, APIs, or databases, enriching its knowledge and abilities beyond its internal training data. This capability dramatically expands the potential of LLMs, allowing them to function not just as isolated models but as integral components of broader, more intelligent systems.

The significance of prompt chaining extends beyond simple problem-solving. It serves as a foundational technique for building sophisticated AI agents. These agents can utilize prompt chains to autonomously plan, reason, and act in dynamic environments. By strategically structuring the sequence of prompts, an agent can engage in tasks requiring multi-step reasoning, planning, and decision-making. Such agent workflows can mimic human thought processes more closely, allowing for more natural and effective interactions with complex domains and systems.

### Limitations of Single Prompts

For multifaceted tasks, using a single, complex prompt for an LLM can be inefficient, causing the model to struggle with constraints and instructions, potentially leading to:

- **Instruction neglect**: where parts of the prompt are overlooked
- **Contextual drift**: where the model loses track of the initial context
- **Error propagation**: where early errors amplify
- **Context window issues**: where the model gets insufficient information to respond back
- **Hallucination**: where the cognitive load increases the chance of incorrect information

For example, a query asking to analyze a market research report, summarize findings, identify trends with data points, and draft an email risks failure as the model might summarize well but fail to extract data or draft an email properly.

### Enhanced Reliability Through Sequential Decomposition

Prompt chaining addresses these challenges by breaking the complex task into a focused, sequential workflow, which significantly improves reliability and control. Given the example above, a pipeline or chained approach can be described as follows:

1. **Initial Prompt (Summarization)**: "Summarize the key findings of the following market research report: [text]." The model's sole focus is summarization, increasing the accuracy of this initial step.

2. **Second Prompt (Trend Identification)**: "Using the summary, identify the top three emerging trends and extract the specific data points that support each trend: [output from step 1]." This prompt is now more constrained and builds directly upon a validated output.

3. **Third Prompt (Email Composition)**: "Draft a concise email to the marketing team that outlines the following trends and their supporting data: [output from step 2]."

This decomposition allows for more granular control over the process. Each step is simpler and less ambiguous, which reduces the cognitive load on the model and leads to a more accurate and reliable final output. This modularity is analogous to a computational pipeline where each function performs a specific operation before passing its result to the next.

### The Role of Structured Output

The reliability of a prompt chain is highly dependent on the integrity of the data passed between steps. If the output of one prompt is ambiguous or poorly formatted, the subsequent prompt may fail due to faulty input. To mitigate this, specifying a structured output format, such as JSON or XML, is crucial.

For example, the output from the trend identification step could be formatted as a JSON object:

```json
{
 "trends": [
   {
     "trend_name": "AI-Powered Personalization",
     "supporting_data": "73% of consumers prefer to do business with brands that use personal information to make their shopping experiences more relevant."
   },
   {
     "trend_name": "Sustainable and Ethical Brands",
     "supporting_data": "Sales of products with ESG-related claims grew 28% over the last five years, compared to 20% for products without."
   }
 ]
}
```

This structured format ensures that the data is machine-readable and can be precisely parsed and inserted into the next prompt without ambiguity. This practice minimizes errors that can arise from interpreting natural language and is a key component in building robust, multi-step LLM-based systems.

## Practical Applications & Use Cases

Prompt chaining is a versatile pattern applicable in a wide range of scenarios when building agentic systems. Its core utility lies in breaking down complex problems into sequential, manageable steps. Here are several practical applications and use cases:

### 1. Information Processing Workflows

Many tasks involve processing raw information through multiple transformations. For instance, summarizing a document, extracting key entities, and then using those entities to query a database or generate a report. A prompt chain could look like:

- **Prompt 1**: Extract text content from a given URL or document.
- **Prompt 2**: Summarize the cleaned text.
- **Prompt 3**: Extract specific entities (e.g., names, dates, locations) from the summary or original text.
- **Prompt 4**: Use the entities to search an internal knowledge base.
- **Prompt 5**: Generate a final report incorporating the summary, entities, and search results.

### 2. Complex Query Answering

Answering complex questions that require multiple steps of reasoning or information retrieval is a prime use case. For example, "What were the main causes of the stock market crash in 1929, and how did government policy respond?"

- **Prompt 1**: Identify the core sub-questions in the user's query (causes of crash, government response).
- **Prompt 2**: Research or retrieve information specifically about the causes of the 1929 crash.
- **Prompt 3**: Research or retrieve information specifically about the government's policy response to the 1929 stock market crash.
- **Prompt 4**: Synthesize the information from steps 2 and 3 into a coherent answer to the original query.

### 3. Data Extraction and Transformation

The conversion of unstructured text into a structured format is typically achieved through an iterative process:

- **Prompt 1**: Attempt to extract specific fields (e.g., name, address, amount) from an invoice document.
- **Processing**: Check if all required fields were extracted and if they meet format requirements.
- **Prompt 2 (Conditional)**: If fields are missing or malformed, craft a new prompt asking the model to specifically find the missing/malformed information.
- **Processing**: Validate the results again. Repeat if necessary.
- **Output**: Provide the extracted, validated structured data.

### 4. Content Generation Workflows

The composition of complex content is a procedural task that is typically decomposed into distinct phases:

- **Prompt 1**: Generate 5 topic ideas based on a user's general interest.
- **Processing**: Allow the user to select one idea or automatically choose the best one.
- **Prompt 2**: Based on the selected topic, generate a detailed outline.
- **Prompt 3**: Write a draft section based on the first point in the outline.
- **Prompt 4**: Write a draft section based on the second point in the outline, providing the previous section for context.
- **Prompt 5**: Review and refine the complete draft for coherence, tone, and grammar.

### 5. Code Generation and Refinement

The generation of functional code is typically a multi-stage process:

- **Prompt 1**: Understand the user's request for a code function. Generate pseudocode or an outline.
- **Prompt 2**: Write the initial code draft based on the outline.
- **Prompt 3**: Identify potential errors or areas for improvement in the code.
- **Prompt 4**: Rewrite or refine the code based on the identified issues.
- **Prompt 5**: Add documentation or test cases.

## Hands-On Code Example

Implementing prompt chaining ranges from direct, sequential function calls within a script to the utilization of specialized frameworks designed to manage control flow, state, and component integration. The following code implements a two-step prompt chain that functions as a data processing pipeline:

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the Language Model
llm = ChatOpenAI(temperature=0)

# --- Prompt 1: Extract Information ---
prompt_extract = ChatPromptTemplate.from_template(
   "Extract the technical specifications from the following text:\n\n{text_input}"
)

# --- Prompt 2: Transform to JSON ---
prompt_transform = ChatPromptTemplate.from_template(
   "Transform the following specifications into a JSON object with 'cpu', 'memory', and 'storage' as keys:\n\n{specifications}"
)

# --- Build the Chain using LCEL ---
extraction_chain = prompt_extract | llm | StrOutputParser()

# The full chain passes the output of the extraction chain into the 'specifications'
# variable for the transformation prompt.
full_chain = (
   {"specifications": extraction_chain}
   | prompt_transform
   | llm
   | StrOutputParser()
)

# --- Run the Chain ---
input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD."

# Execute the chain with the input text dictionary.
final_result = full_chain.invoke({"text_input": input_text})

print("\n--- Final JSON Output ---")
print(final_result)
```

This code demonstrates how to use LangChain to process text through two separate prompts: one to extract technical specifications and another to format these specifications into a JSON object.

## Context Engineering and Prompt Engineering

Context Engineering is the systematic discipline of designing, constructing, and delivering a complete informational environment to an AI model prior to token generation. This methodology asserts that the quality of a model's output is less dependent on the model's architecture itself and more on the richness of the context provided.

It represents a significant evolution from traditional prompt engineering, which focuses primarily on optimizing the phrasing of a user's immediate query. Context Engineering expands this scope to include several layers of information, such as:

- **System prompts**: Foundational instructions defining the AI's operational parameters
- **External data**: Retrieved documents from knowledge bases
- **Tool outputs**: Results from external API calls for real-time data
- **Implicit data**: User identity, interaction history, and environmental state

## Key Takeaways

- **Prompt Chaining** breaks down complex tasks into a sequence of smaller, focused steps
- Each step in a chain involves an LLM call or processing logic, using the output of the previous step as input
- This pattern improves the reliability and manageability of complex interactions with language models
- Frameworks like LangChain/LangGraph and Google ADK provide robust tools to define, manage, and execute these multi-step sequences
- **Context Engineering** is crucial for building comprehensive informational environments that enable advanced agentic performance

## Visual Summary

*Prompt Chaining Pattern: Agents receive a series of prompts from the user, with the output of each agent serving as the input for the next in the chain.*

## When to Use This Pattern

**Use this pattern when:**
- A task is too complex for a single prompt
- The task involves multiple distinct processing stages
- You need interaction with external tools between steps
- Building agentic systems that need to perform multi-step reasoning and maintain state

## Conclusion

By deconstructing complex problems into a sequence of simpler, more manageable sub-tasks, prompt chaining provides a robust framework for guiding large language models. This "divide-and-conquer" strategy significantly enhances the reliability and control of the output by focusing the model on one specific operation at a time. As a foundational pattern, it enables the development of sophisticated AI agents capable of multi-step reasoning, tool integration, and state management.

## References

1. LangChain Documentation on LCEL: https://python.langchain.com/v0.2/docs/core_modules/expression_language/
2. LangGraph Documentation: https://langchain-ai.github.io/langgraph/
3. Prompt Engineering Guide - Chaining Prompts: https://www.promptingguide.ai/techniques/chaining
4. OpenAI API Documentation: https://platform.openai.com/docs/guides/gpt/prompting
5. Crew AI Documentation: https://docs.crewai.com/
6. Google AI for Developers: https://cloud.google.com/discover/what-is-prompt-engineering?hl=en
7. Vertex Prompt Optimizer: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-optimizer
- Testing and validation can be performed at the component level

## Implementation

### Basic Chain Structure
```python
def prompt_chain(initial_input):
    # Step 1: Analysis
    analysis = call_llm(analysis_prompt, initial_input)

    # Step 2: Planning
    plan = call_llm(planning_prompt, analysis)

    # Step 3: Execution
    result = call_llm(execution_prompt, plan)

    return result
```

### Advanced Chain Management
- Implement error handling and retry logic
- Add logging and monitoring at each step
- Include conditional branching based on intermediate results
- Store intermediate states for debugging and analysis

## Code Examples

### Example 1: Document Analysis Chain
```python
class DocumentAnalysisChain:
    def __init__(self):
        self.summarizer_prompt = "Summarize the following document: {document}"
        self.extractor_prompt = "Extract key entities from: {summary}"
        self.classifier_prompt = "Classify the document type based on: {entities}"

    def analyze(self, document):
        # Step 1: Summarize
        summary = self.llm_call(self.summarizer_prompt, document=document)

        # Step 2: Extract entities
        entities = self.llm_call(self.extractor_prompt, summary=summary)

        # Step 3: Classify
        classification = self.llm_call(self.classifier_prompt, entities=entities)

        return {
            'summary': summary,
            'entities': entities,
            'classification': classification
        }
```

### Example 2: Creative Writing Chain
```python
def creative_writing_chain(topic, style):
    # Generate outline
    outline_prompt = f"Create an outline for a {style} story about {topic}"
    outline = generate_response(outline_prompt)

    # Develop characters
    character_prompt = f"Based on this outline: {outline}, create detailed characters"
    characters = generate_response(character_prompt)

    # Write story
    story_prompt = f"Write a {style} story using outline: {outline} and characters: {characters}"
    story = generate_response(story_prompt)

    return story
```

## Best Practices

### Design Principles
- **Single Responsibility**: Each prompt should have one clear purpose
- **Clear Interfaces**: Define explicit input/output formats between prompts
- **Error Boundaries**: Implement proper error handling at each step
- **Logging**: Maintain comprehensive logs of the chain execution

### Optimization Strategies
- **Prompt Engineering**: Optimize each prompt individually for its specific task
- **Caching**: Cache intermediate results where appropriate
- **Parallel Execution**: Identify opportunities for parallel processing
- **Resource Management**: Monitor token usage and API costs

### Testing and Validation
- Test each prompt in isolation before integration
- Create comprehensive test suites for the entire chain
- Monitor chain performance and accuracy over time
- Implement A/B testing for prompt variations

## Common Pitfalls

### Information Loss
- **Problem**: Important context gets lost between chain steps
- **Solution**: Implement proper state management and context preservation
- **Mitigation**: Include relevant context in each prompt

### Chain Brittleness
- **Problem**: Failure in one step breaks the entire chain
- **Solution**: Implement robust error handling and fallback mechanisms
- **Mitigation**: Design redundant pathways for critical operations

### Over-Engineering
- **Problem**: Creating unnecessarily complex chains for simple tasks
- **Solution**: Start simple and add complexity only when needed
- **Mitigation**: Regular review and simplification of chain logic

### Cost Accumulation
- **Problem**: Multiple API calls can become expensive
- **Solution**: Optimize prompt efficiency and implement caching
- **Mitigation**: Monitor costs and implement usage limits

### Context Window Limitations
- **Problem**: Accumulated context exceeds model limits
- **Solution**: Implement context pruning and summarization strategies
- **Mitigation**: Design chains with context window constraints in mind

## Conclusion

Prompt chaining provides a robust foundation for building sophisticated agentic AI systems. By breaking complex tasks into manageable sequential steps, this pattern enables developers to create more reliable, maintainable, and interpretable AI workflows. Success with prompt chaining requires careful attention to design principles, proper error handling, and continuous optimization of individual chain components.