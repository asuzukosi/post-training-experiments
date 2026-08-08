from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.create_draft import GmailCreateDraft
from langchain.tools import tool


class CreateDraftTool:
    @tool("Create Draft")
    def create_draft(data):
        """
        useful to create an email draft.
        the input should be pipe-separated text of length three:
        who to send to, subject, and message.
        for example, `lorem@ipsum.com|nice to meet you|hey it was great to meet you.`.
        """
        email, subject, message = data.split("|")
        gmail = GmailToolkit()
        draft = GmailCreateDraft(api_resource=gmail.api_resource)
        result = draft(
            {
                "to": [email],
                "subject": subject,
                "message": message,
            }
        )
        return f"\ndraft created: {result}\n"
