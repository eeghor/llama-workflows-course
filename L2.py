import random
import asyncio
import nest_asyncio

from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Event,
    Workflow,
    step,
    Context,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_parse import LlamaParse
import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex

from dotenv import load_dotenv

load_dotenv()
# we need to be able to run
# asyncio operations nested within other asyncio contexts
nest_asyncio.apply()

api_key = {
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "llama_cloud": os.getenv("LLAMA_CLOUD_API_KEY"),
}

# Events are simple data classes that pass information
# between workflow steps

selected_llm = Anthropic(model="claude-3-7-sonnet-latest", api_key=api_key["anthropic"])


class FirstEvent(Event):
    first_event_output: str


class SecondEvent(Event):
    second_output: str
    response: str


class TextEvent(Event):
    chunk_of_text_event_gets: str


class ProgressEvent(Event):
    msg: str


class LoopEvent(Event):
    loop_output: str


class BranchA1Event(Event):
    payload: str


class BranchA2Event(Event):
    payload: str


class BranchB1Event(Event):
    payload: str


class BranchB2Event(Event):
    payload: str


class StepTwoEvent(Event):
    query: str


class StepThreeEvent(Event):
    result: str


class StepAEvent(Event):
    query: str


class StepACompleteEvent(Event):
    result: str


class StepBEvent(Event):
    query: str


class StepBCompleteEvent(Event):
    result: str


class StepCEvent(Event):
    query: str


class StepCCompleteEvent(Event):
    result: str


class MyWorkflow(Workflow):
    """
    Define the sequence of steps and logic for the workflow.
    """

    # Steps: methods decorated with @step that process events and emit new events
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        return FirstEvent(first_event_output="First step complete.")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        llm = selected_llm
        generator = await llm.astream_complete(
            "Please give me the first 50 words of Moby Dick, a book in the public domain."
        )
        async for response in generator:
            # Allow the workflow to stream this piece of response
            ctx.write_event_to_stream(
                TextEvent(chunk_of_text_event_gets=response.chunk_of_text_event_gets)
            )
        return SecondEvent(
            second_output="Second step complete, full response attached",
            response=str(response),
        )

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step three is happening"))
        return StopEvent(result="Workflow complete.")


class BranchWorkflow(Workflow):
    @step
    async def start(self, ev: StartEvent) -> BranchA1Event | BranchB1Event:
        if random.randint(0, 1) == 0:
            print("Go to branch A")
            return BranchA1Event(payload="Branch A - Event 1")
        else:
            print("Go to branch B")
            return BranchB1Event(payload="Branch B - Event 1")

    @step
    async def step_a1(self, ev: BranchA1Event) -> BranchA2Event:
        print(ev.payload)
        return BranchA2Event(payload="Branch A - Event 2")

    @step
    async def step_a2(self, ev: BranchA2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch A complete.")

    @step
    async def step_b1(self, ev: BranchB1Event) -> BranchB2Event:
        print(ev.payload)
        return BranchB2Event(payload="Branch B - Event 2")

    @step
    async def step_b2(self, ev: BranchB2Event) -> StopEvent:
        print(ev.payload)
        return StopEvent(result="Branch B complete.")


class ParallelFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> StepTwoEvent:
        ctx.send_event(StepTwoEvent(query="Query 1"))
        ctx.send_event(StepTwoEvent(query="Query 2"))
        ctx.send_event(StepTwoEvent(query="Query 3"))

    @step(num_workers=4)
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
        print("Running slow query ", ev.query)
        await asyncio.sleep(random.randint(1, 5))

        return StopEvent(result=ev.query)


class ConcurrentFlow(Workflow):
    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> StepAEvent | StepBEvent | StepCEvent:
        ctx.send_event(StepAEvent(query="Query 1"))
        ctx.send_event(StepBEvent(query="Query 2"))
        ctx.send_event(StepCEvent(query="Query 3"))

    @step
    async def step_a(self, ctx: Context, ev: StepAEvent) -> StepACompleteEvent:
        print("Doing something A-ish")
        return StepACompleteEvent(result=ev.query)

    @step
    async def step_b(self, ctx: Context, ev: StepBEvent) -> StepBCompleteEvent:
        print("Doing something B-ish")
        return StepBCompleteEvent(result=ev.query)

    @step
    async def step_c(self, ctx: Context, ev: StepCEvent) -> StepCCompleteEvent:
        print("Doing something C-ish")
        return StepCCompleteEvent(result=ev.query)

    @step
    async def step_three(
        self,
        ctx: Context,
        ev: StepACompleteEvent | StepBCompleteEvent | StepCCompleteEvent,
    ) -> StopEvent:
        print("Received event ", ev.result)

        # wait until we receive 3 events
        events = ctx.collect_events(
            ev,
            [StepCCompleteEvent, StepACompleteEvent, StepBCompleteEvent],
        )
        if events is None:
            return None

        # do something with all 3 results together
        print("All events received: ", events)
        return StopEvent(result="Done")


async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)

    # and now runs the remaining workflows
    w = BranchWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)

    w = ParallelFlow(timeout=10, verbose=False)
    result = await w.run()
    print(result)

    w = ConcurrentFlow(timeout=10, verbose=False)
    result = await w.run()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

    documents = LlamaParse(
        api_key=api_key["llama_cloud"],
        base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
        result_type="markdown",
        content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers",
    ).load_data(
        "data/fake_resume.pdf",
    )

    print(documents[2].text)

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=OpenAIEmbedding(
            model_name="text-embedding-3-small", api_key=api_key["openai"]
        ),
    )
    print("Index created")

    query_engine = index.as_query_engine(
        llm=selected_llm,
        similarity_top_k=5,
    )

    print("Query engine created")

    response = query_engine.query(
        "What is this person's name and what was their most recent job?"
    )
    print(response)

    # store index to disk
    storage_dir = "./storage"

    index.storage_context.persist(persist_dir=storage_dir)
    print("index made persistent")

    # Check if the index is stored on disk
    if os.path.exists(storage_dir):
        # Load the index from disk
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        restored_index = load_index_from_storage(storage_context)
        print("Index restored from disk.")
    else:
        print("Index not found on disk.")

    print("Querying the restored index")
    response = restored_index.as_query_engine().query(
        "What is this person's name and what was their most recent job?"
    )
    print(response)

    def query_resume(q: str) -> str:
        """Answers questions about a specific resume."""
        # we're using the query engine we already created above
        response = query_engine.query(
            f"This is a question about the specific resume we have in our database: {q}"
        )
        return response.response

    resume_tool = FunctionTool.from_defaults(fn=query_resume)

    agent = FunctionCallingAgent.from_tools(
        tools=[resume_tool],
        llm=selected_llm,
        verbose=True,
    )
    print("Agent set up. Asking a question about the resume")

    response = agent.chat("How many years of experience does the applicant have?")
    print(response)

    class QueryEvent(Event):
        query: str

    class RAGWorkflow(Workflow):
        storage_dir = "./storage"
        llm: OpenAI
        query_engine: VectorStoreIndex

        # the first step will be setup
        @step
        async def set_up(self, ctx: Context, ev: StartEvent) -> QueryEvent:
            if not ev.resume_file:
                raise ValueError("No resume file provided")

            # define an LLM to work with
            self.llm = OpenAI(model="gpt-4o-mini", api_key=api_key["openai"])

            # ingest the data and set up the query engine
            if os.path.exists(self.storage_dir):
                # you've already ingested your documents
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.storage_dir
                )
                index = load_index_from_storage(storage_context)
            else:
                # parse and load your documents
                documents = LlamaParse(
                    result_type="markdown",
                    content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers",
                ).load_data(ev.resume_file)
                # embed and index the documents
                index = VectorStoreIndex.from_documents(
                    documents,
                    embed_model=OpenAIEmbedding(
                        model_name="text-embedding-3-small", api_key=api_key["openai"]
                    ),
                )
                index.storage_context.persist(persist_dir=self.storage_dir)

            # either way, create a query engine
            self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

            # now fire off a query event to trigger the next step
            return QueryEvent(query=ev.query)

        # the second step will be to ask a question and return a result immediately
        @step
        async def ask_question(self, ctx: Context, ev: QueryEvent) -> StopEvent:
            response = self.query_engine.query(
                f"This is a question about the specific resume we have in our database: {ev.query}"
            )
            return StopEvent(result=response.response)
