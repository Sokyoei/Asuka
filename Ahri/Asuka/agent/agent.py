from uuid import uuid4

from langchain_classic.memory import ConversationTokenBufferMemory
from langchain_classic.memory.chat_memory import BaseChatMemory
from langchain_classic.output_parsers import OutputFixingParser
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, render_text_description
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from Ahri.Asuka.agent.action import Action
from Ahri.Asuka.agent.utils import THOUGHT_COLOR, ColoredPrintHandler


class Agent:

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        work_dir: str,
        main_prompt_file: str,
        max_thought_steps: int = 10,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.work_dir = work_dir
        self.max_thought_steps = max_thought_steps

        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(
            parser=self.output_parser, llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, model_kwargs={"seed": 18})
        )

        self.main_prompt_file = main_prompt_file

        self._init_prompt_templates()
        self._init_chains()

        self.verbose_handler = ColoredPrintHandler(color=THOUGHT_COLOR)

    def _init_prompt_templates(self):
        with open(self.main_prompt_file, encoding="utf8") as f:
            self.prompt = ChatPromptTemplate.from_messages(
                [MessagesPlaceholder(variable_name="chat_history"), HumanMessagePromptTemplate.from_template(f.read())]
            ).partial(
                work_dir=self.work_dir,
                tools=render_text_description(self.tools),
                tool_names=",".join([tool.name for tool in self.tools]),
                format_instructions=self.output_parser.get_format_instructions(),
            )

    def _init_chains(self):
        self.main_chains = self.prompt | self.llm | StrOutputParser()

    def _step(self, task, short_term_memory, chat_history, verbose=False) -> tuple[Action, str]:
        response = ""
        for s in self.main_chains.stream(
            {
                "input": task,
                "agent_scratchpad": self._format_short_term_memory(short_term_memory),
                "chat_history": chat_history,
            },
            config={"callbacks": [self.verbose_handler] if verbose else []},
        ):
            response += s

        action = self.robust_parser.parse(response)
        return action, response

    def _format_short_term_memory(self, memory: BaseChatMemory) -> str:
        messages = memory.chat_memory.messages
        string_messages = [str(messages[i].content) for i in range(1, len(messages))]
        return "\n".join(string_messages)

    def _find_tool(self, name) -> BaseTool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def _exec_action(self, action):
        tool = self._find_tool(action.name)
        if tool is None:
            observation = (
                f"ERROR: 找不到工具或指令 {action.name}, 请从提供的工具或指令列表中选择，请确保按对顶格式输出。"
            )
        else:
            try:
                observation = tool.run(action.args)
            except ValidationError as e:
                observation = f"ERROR: ValidationError in args: {e!s}, args: {action.args}"
            except Exception as e:
                observation = f"ERROR: {e=}, args: {action.args}"
        return observation

    def run(self, task: str, chat_history: ChatMessageHistory, verbose: bool = False):
        # 初始化短时记忆
        short_term_memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=4000)
        # 思考步数
        thought_step = 0
        reply = ""

        while thought_step < self.max_thought_steps:
            if verbose:
                self.verbose_handler.on_thought_start(thought_step)

            # 执行一步思考
            action, response = self._step(
                task=task, short_term_memory=short_term_memory, chat_history=chat_history, verbose=verbose
            )

            # 结束指令，执行最后一步
            if action.name == "FINISH":
                reply = self._exec_action(action)
                break

            observation = self._exec_action(action)
            if verbose:
                self.verbose_handler.on_tool_end(observation, run_id=uuid4())

            # 更新短时记忆
            short_term_memory.save_context(inputs={"input": response}, outputs={"output": "\n返回结果\n" + observation})

            thought_step += 1

        if thought_step >= self.max_thought_steps:
            reply = "抱歉，我没能完成您的任务。"

        chat_history.add_user_message(task)
        chat_history.add_ai_message(reply)
        return reply
