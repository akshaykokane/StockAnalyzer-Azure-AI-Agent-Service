using Azure;
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.AzureAI;
using Microsoft.SemanticKernel.Agents.Chat;
using Microsoft.SemanticKernel.ChatCompletion;
using System;
using System.Threading.Tasks;
using Agent = Azure.AI.Projects.Agent;

namespace AzureAIAgentServiceDemo
{
    class Program
    {
        private AIProjectClient Client;
        private Kernel kernel;

        static async Task Main(string[] args)
        {
            Console.WriteLine("Welcome to My Console Application!");

            // Handle command-line arguments
            if (args.Length > 0)
            {
                Console.WriteLine("Arguments received:");
                foreach (var arg in args)
                {
                    Console.WriteLine($"- {arg}");
                }
            }
            else
            {
                Console.WriteLine("No arguments passed.");
            }

            // Execute the application logic
            await RunApplication();

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        static async Task RunApplication()
        {
            Console.WriteLine("Running application logic...");

            var builder = Kernel.CreateBuilder();
            builder.AddAzureOpenAIChatCompletion("GPT4ov1", "https://<deploymentname>.openai.azure.com", "<OpenAIKey>");
            Kernel kernel = builder.Build();

            var credential = new DefaultAzureCredential(new DefaultAzureCredentialOptions
            {
                ExcludeInteractiveBrowserCredential = false,
            });

            string connectionString = "<AIProjectConnectionString>";
            AgentsClient client = new(connectionString, credential);

            // Initialize Stock Expert Agent
            Agent stockExpertAgent = await GetAgentById(client, "asst_<ID>");
            AzureAIAgent stockExpert = new(stockExpertAgent, client) { Kernel = kernel };

            // Initialize Investor Advisor Agent
            Agent investorAdvisorAgent = await GetAgentById(client, "asst_<ID>");
            AzureAIAgent investorAdvisor = new(investorAdvisorAgent, client) { Kernel = kernel };

            // Configure agent group chat
            var agentGroupChat = ConfigureAgentGroupChat(stockExpert, investorAdvisor, kernel);

            // Add user query and invoke chat
            agentGroupChat.AddChatMessage(new ChatMessageContent(AuthorRole.User, "I am interested in buying MSFT or AMZ stocks. Provide me investment advice based on my portfolio"));
            await foreach (var content in agentGroupChat.InvokeAsync())
            {
                Console.WriteLine($"<b>#{content.Role}</b> - <i>{content.AuthorName ?? "*"}</i>: \"{content.Content}\"");
            }
        }

        private static async Task<Agent> GetAgentById(AgentsClient client, string agentId)
        {
            Response<Agent> agentResponse = await client.GetAgentAsync(agentId);
            return agentResponse.Value;
        }

        private static AgentGroupChat ConfigureAgentGroupChat(AzureAIAgent stockExpert, AzureAIAgent investorAdvisor, Kernel kernel)
        {
            var selectionStrategy = new KernelFunctionSelectionStrategy(GetSelectionFunction(investorAdvisor.Name, stockExpert.Name), kernel)
            {
                InitialAgent = stockExpert,
                HistoryVariableName = "history",
                HistoryReducer = new ChatHistoryTruncationReducer(10),
            };

            var terminationStrategy = new KernelFunctionTerminationStrategy(GetTerminationStrategy(), kernel)
            {
                ResultParser = result => result.GetValue<string>()?.Contains("done", StringComparison.OrdinalIgnoreCase) ?? false,
                HistoryVariableName = "history",
                HistoryReducer = new ChatHistoryTruncationReducer(10),
                MaximumIterations = 10,
            };

            return new AgentGroupChat(stockExpert, investorAdvisor)
            {
                ExecutionSettings = new()
                {
                    SelectionStrategy = selectionStrategy,
                    TerminationStrategy = terminationStrategy,
                }
            };
        }

        private static KernelFunction GetSelectionFunction(string investorAdvisor, string stockExpert)
        {
            return AgentGroupChat.CreatePromptFunctionForStrategy(
                $$$"""
                Determine which participant takes the next turn in a conversation.
                State only the name of the participant to take the next turn.
                No participant should take more than one turn in a row.

                Choose only from these participants:
                - {stockExpert}
                - {investorAdvisor}

                History:
                {$history}
                """,
                safeParameterNames: "history");
        }

        private static KernelFunction GetTerminationStrategy()
        {
            return AgentGroupChat.CreatePromptFunctionForStrategy(
                $$$"""
                Determine if the question has been answered and investment advisor agent provided the personalized advice. If yes, just reply "done".

                History:
                {$history}
                """,
                safeParameterNames: "history");
        }
    }
}
