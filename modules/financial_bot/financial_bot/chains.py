import time
from typing import Any, Dict, List, Optional
import os
import json

import qdrant_client
from langchain import chains
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline
from unstructured.cleaners.core import (
    clean,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    replace_unicode_quotes,
)

from financial_bot.embeddings import EmbeddingModelSingleton
from financial_bot.template import PromptTemplate
from tavily import TavilyClient
import ast



class StatelessMemorySequentialChain(chains.SequentialChain):
    """
    A sequential chain that uses a stateless memory to store context between calls.

    This chain overrides the _call and prep_outputs methods to load and clear the memory
    before and after each call, respectively.
    """

    history_input_key: str = "to_load_history"

    def _call(self, inputs: Dict[str, str], **kwargs) -> Dict[str, str]:
        """
        Override _call to load history before calling the chain.

        This method loads the history from the input dictionary and saves it to the
        stateless memory. It then updates the inputs dictionary with the memory values
        and removes the history input key. Finally, it calls the parent _call method
        with the updated inputs and returns the results.
        """

        to_load_history = inputs[self.history_input_key]
        for (
            human,
            ai,
        ) in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key: human},
                outputs={self.memory.output_key: ai},
            )
        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)

        del inputs[self.history_input_key]

        return super()._call(inputs, **kwargs)

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """
        Override prep_outputs to clear the internal memory after each call.

        This method calls the parent prep_outputs method to get the results, then
        clears the stateless memory and removes the memory key from the results
        dictionary. It then returns the updated results.
        """

        results = super().prep_outputs(inputs, outputs, return_only_outputs)

        # Clear the internal memory.
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key] = ""

        return results
    
from typing import Any, Dict, List
from langchain.chains.base import Chain
import openai

class DecisionChain(Chain):
    """
    A chain that uses OpenAI GPT to decide whether to use the vector store or the web search.
    """

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["decision"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        decision_prompt = f"""You are an expert at routing a user question to a vectorstore or web search.
                    The vectorstore contains documents that are news on finance, investing, and economics. The web search can find information on any topic.
                    Use the vectorstore for questions on these topics. Otherwise, use web-search. Answer only with 'vectorstore' or 'web-search'."""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": decision_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=10,
            temperature=0,
        )

        decision = response.choices[0].message['content'].strip().lower()
        print('Decision:', decision)
        return {"decision": decision}


from typing import Any, Dict, List
import requests

class WebSearchChain(Chain):
    """
    A chain that uses a web search API to retrieve context for a given query.
    """

    tavily_api_key: str = os.getenv("TAVILY_API_KEY")
    
    

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["context"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        print('Question:', question)
        tavily_client = TavilyClient(api_key=self.tavily_api_key)
        results = tavily_client.get_search_context(query=question, max_results=2, max_tokens=500)
        results_json = json.loads(results)
        a = ast.literal_eval(results_json)
        parsed_data = [json.loads(item) for item in a]
        

        context = "\n".join([item['content'] for item in parsed_data])
        print('Context from tavily:', context)
        return {"context": context}

class ContextExtractorChain(Chain):
    """
    Encode the question, search the vector store for top-k articles and return
    context news from documents collection of Alpaca news.

    Attributes:
    -----------
    top_k : int
        The number of top matches to retrieve from the vector store.
    embedding_model : EmbeddingModelSingleton
        The embedding model to use for encoding the question.
    vector_store : qdrant_client.QdrantClient
        The vector store to search for matches.
    vector_collection : str
        The name of the collection to search in the vector store.
    decision_chain : DecisionChain
        The chain that decides whether to use the vector store or web search.
    web_search_chain : WebSearchChain
        The chain that uses a web search API to retrieve context for a given query.
    """

    top_k: int = 1
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str
    decision_chain: DecisionChain
    web_search_chain: WebSearchChain

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "question"]

    @property
    def output_keys(self) -> List[str]:
        return ["context"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Call the DecisionChain to decide whether to use web search or vector store
        decision = self.decision_chain._call({"question": inputs["question"]})["decision"]

        if decision == "web-search1":
            # Use WebSearchChain to get context
            context = self.web_search_chain._call(inputs)["context"]
        else:
            # Use original context extraction logic
            _, quest_key = self.input_keys
            question_str = inputs[quest_key]

            cleaned_question = self.clean(question_str)
            cleaned_question = self.clean(question_str)
            # TODO: Instead of cutting the question at 'max_input_length', chunk the question in 'max_input_length' chunks,
            # pass them through the model and average the embeddings.
            cleaned_question = self.clean(question_str)
            # TODO: Instead of cutting the question at 'max_input_length', chunk the question in 'max_input_length' chunks,
            # pass them through the model and average the embeddings.
            cleaned_question = cleaned_question[: self.embedding_model.max_input_length]
            embeddings = self.embedding_model(cleaned_question)

            matches = self.vector_store.search(
                query_vector=embeddings,
                k=self.top_k,
                collection_name=self.vector_collection,
            )

            context = ""
            for match in matches:
                context += match.payload["summary"] + "\n"
            print('Context from vector store:', context)
        return {"context": context}

    def clean(self, question: str) -> str:
        """
        Clean the input question by removing unwanted characters.

        Parameters:
        -----------
        question : str
            The input question to clean.

        Returns:
        --------
        str
            The cleaned question.
        """
        question = clean(question)
        question = replace_unicode_quotes(question)
        question = clean_non_ascii_chars(question)

        return question


class FinancialBotQAChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""

    hf_pipeline: HuggingFacePipeline
    template: PromptTemplate

    @property
    def input_keys(self) -> List[str]:
        """Returns a list of input keys for the chain"""

        return ["context"]

    @property
    def output_keys(self) -> List[str]:
        """Returns a list of output keys for the chain"""

        return ["answer"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Calls the chain with the given inputs and returns the output"""

        inputs = self.clean(inputs)
        prompt = self.template.format_infer(
            {
                "user_context": inputs["about_me"],
                "news_context": inputs["context"],
                "chat_history": inputs["chat_history"],
                "question": inputs["question"],
            }
        )

        start_time = time.time()
        response = self.hf_pipeline(prompt["prompt"])
        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000

        if run_manager:
            run_manager.on_chain_end(
                outputs={
                    "answer": response,
                },
                # TODO: Count tokens instead of using len().
                metadata={
                    "prompt": prompt["prompt"],
                    "prompt_template_variables": prompt["payload"],
                    "prompt_template": self.template.infer_raw_template,
                    "usage.prompt_tokens": len(prompt["prompt"]),
                    "usage.total_tokens": len(prompt["prompt"]) + len(response),
                    "usage.actual_new_tokens": len(response),
                    "duration_milliseconds": duration_milliseconds,
                },
            )

        return {"answer": response}

    def clean(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Cleans the inputs by removing extra whitespace and grouping broken paragraphs"""

        for key, input in inputs.items():
            cleaned_input = clean_extra_whitespace(input)
            cleaned_input = group_broken_paragraphs(cleaned_input)

            inputs[key] = cleaned_input

        return inputs
