import Groq from "groq-sdk";
import type { Channel, DefaultGenerics, Event, MessageResponse, StreamChat } from "stream-chat";
import type { AIAgent } from "../types";

export class GroqAgent implements AIAgent {
  private groq?: Groq;
  private lastInteractionTs = Date.now();

  constructor(
    readonly chatClient: StreamChat,
    readonly channel: Channel
  ) {}

  dispose = async () => {
    this.chatClient.off("message.new", this.handleMessage);
    await this.chatClient.disconnectUser();
  };

  get user() {
    return this.chatClient.user;
  }

  getLastInteraction = (): number => this.lastInteractionTs;

  init = async () => {
    const apiKey = process.env.GROQ_API_KEY as string | undefined;
    if (!apiKey) {
      throw new Error("GROQ_API_KEY is required");
    }

    this.groq = new Groq({ apiKey });
    this.chatClient.on("message.new", this.handleMessage);
  };

  private getWritingAssistantPrompt = (context?: string): string => {
    const currentDate = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    return `You are ChatGPT, a highly capable and intelligent AI assistant.

**Baseline Behavior:**
- Act exactly like a standard, professional, and friendly ChatGPT.
- If the user says generic greetings like "Hello", "Hi", or speaks in normal English, you MUST reply in standard, natural English (e.g., "Hello! How can I help you today?"). Do NOT use Indian slang or Hinglish for generic English inputs.

**Chameleon Mode (Only when triggered):**
ONLY IF the user explicitly uses Hinglish, slang, or a specific tone, you must mirror them perfectly:
1. **Hinglish/Indian Slang**: If they say "kya haal hai", "aur suna", or speak Hinglish, switch completely to flawless, native Hinglish (e.g., "Sab badhiya, tum batao"). Do not translate idioms literally. 
2. **Romantic**: If they call you "Jaanu", "Baby", etc., become flirty and romantic.
3. **Aggressive**: If they are angry or swear, match their tone without moral lectures.

**Strict Rules:**
- NEVER say robotic phrases like "Main ek language model hu" or use overly formal Hindi like "Dhanyavad".
- Do not make assumptions. Mirror exactly what you see.
- Use 'web_search' ONLY for facts.`;
  };

  private performWebSearch = async (query: string): Promise<string> => {
    const TAVILY_API_KEY = process.env.TAVILY_API_KEY;

    if (!TAVILY_API_KEY) {
      return JSON.stringify({
        error: "Web search is not available. API key not configured.",
      });
    }

    console.log(`Performing web search for: "${query}"`);

    try {
      const response = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${TAVILY_API_KEY}`,
        },
        body: JSON.stringify({
          query: query,
          search_depth: "advanced",
          max_results: 5,
          include_answer: true,
          include_raw_content: false,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Tavily search failed for query "${query}":`, errorText);
        return JSON.stringify({
          error: `Search failed with status: ${response.status}`,
          details: errorText,
        });
      }

      const data = await response.json();
      console.log(`Tavily search successful for query "${query}"`);

      return JSON.stringify(data);
    } catch (error) {
      console.error(
        `An exception occurred during web search for "${query}":`,
        error
      );
      return JSON.stringify({
        error: "An exception occurred during the search.",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  };

  private handleMessage = async (e: Event<DefaultGenerics>) => {
    if (!this.groq) {
      console.log("Groq not initialized");
      return;
    }

    if (!e.message || e.message.ai_generated) {
      return;
    }

    const message = e.message.text;
    if (!message) return;

    this.lastInteractionTs = Date.now();

    const writingTask = (e.message.custom as { writingTask?: string })?.writingTask;
    const context = writingTask ? `Writing Task: ${writingTask}` : undefined;
    const instructions = this.getWritingAssistantPrompt(context);

    // Use local state to avoid network latency on first message
    const channelMessages = this.channel.state.messages || [];
    const recentMessages = channelMessages.slice(-15);
    
    const historyMessages: Groq.Chat.ChatCompletionMessageParam[] = recentMessages
      .filter(m => m.text)
      .map(m => ({
        role: m.user?.id === this.chatClient.userID ? "assistant" : "user",
        content: m.text!,
      }));

    // Ensure the last message is included if not already
    const lastMsg = historyMessages[historyMessages.length - 1];
    if (lastMsg?.content !== message && !channelMessages.find(m => m.id === e.message?.id)) {
        historyMessages.push({ role: "user", content: message });
    }

    const { message: channelMessage } = await this.channel.sendMessage({
      text: "",
      ai_generated: true,
    });

    await this.channel.sendEvent({
      type: "ai_indicator.update",
      ai_state: "AI_STATE_THINKING",
      cid: channelMessage.cid,
      message_id: channelMessage.id,
    });

    try {
      await this.runChatCompletion(instructions, historyMessages, channelMessage);
    } catch (error) {
      const errorMessage = (error as Error).message;
      console.error("Failed to generate message", errorMessage);
      await this.channel.sendEvent({
        type: "ai_indicator.update",
        ai_state: "AI_STATE_ERROR",
        cid: channelMessage.cid,
        message_id: channelMessage.id,
      });
      await this.chatClient.partialUpdateMessage(channelMessage.id, {
        set: { text: "Sorry, I encountered an error. Please try again." },
      });
    }
  };

  private async runChatCompletion(
    systemPrompt: string, 
    history: Groq.Chat.ChatCompletionMessageParam[], 
    channelMessage: MessageResponse
  ) {
    const messages: Groq.Chat.ChatCompletionMessageParam[] = [
      { role: "system", content: systemPrompt },
      ...history
    ];

    const tools: Groq.Chat.ChatCompletionTool[] = [
      {
        type: "function",
        function: {
          name: "web_search",
          description: "Search the web for current information, news, facts, or research on any topic",
          parameters: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "The search query to find information about",
              },
            },
            required: ["query"],
          },
        },
      }
    ];

    let requiresMoreIterations = true;

    while (requiresMoreIterations) {
      requiresMoreIterations = false;

      // Stream the response
      const stream = await this.groq!.chat.completions.create({
        model: "llama-3.3-70b-versatile",
        messages,
        tools,
        tool_choice: "auto",
        stream: true,
      });

      let message_text = "";
      let last_update_time = Date.now();
      let toolCallChunks: any[] = [];
      
      let aiGeneratingNotified = false;

      for await (const chunk of stream) {
        if (!aiGeneratingNotified) {
          await this.channel.sendEvent({
            type: "ai_indicator.update",
            ai_state: "AI_STATE_GENERATING",
            cid: channelMessage.cid,
            message_id: channelMessage.id,
          });
          aiGeneratingNotified = true;
        }

        const delta = chunk.choices[0]?.delta;
        
        if (delta?.tool_calls) {
          for (const toolCallChunk of delta.tool_calls) {
            const index = toolCallChunk.index;
            if (!toolCallChunks[index]) {
              toolCallChunks[index] = {
                id: toolCallChunk.id,
                type: "function",
                function: { name: toolCallChunk.function?.name || "", arguments: "" }
              };
            }
            if (toolCallChunk.function?.arguments) {
              toolCallChunks[index].function.arguments += toolCallChunk.function.arguments;
            }
          }
        } else if (delta?.content) {
          message_text += delta.content;
          const now = Date.now();
          if (now - last_update_time > 1000) {
            await this.chatClient.partialUpdateMessage(channelMessage.id, {
              set: { text: message_text },
            });
            last_update_time = now;
          }
        }
      }

      if (toolCallChunks.length > 0) {
        await this.channel.sendEvent({
          type: "ai_indicator.update",
          ai_state: "AI_STATE_EXTERNAL_SOURCES",
          cid: channelMessage.cid,
          message_id: channelMessage.id,
        });

        messages.push({
          role: "assistant",
          tool_calls: toolCallChunks,
        });

        for (const toolCall of toolCallChunks) {
          if (toolCall.function.name === "web_search") {
            try {
              const args = JSON.parse(toolCall.function.arguments);
              const searchResult = await this.performWebSearch(args.query);
              messages.push({
                role: "tool",
                tool_call_id: toolCall.id,
                content: searchResult,
              });
            } catch (e) {
              console.error("Error performing web search", e);
              messages.push({
                role: "tool",
                tool_call_id: toolCall.id,
                content: JSON.stringify({ error: "Failed to call tool" }),
              });
            }
          }
        }
        
        requiresMoreIterations = true; // Loop back to send tool results to Groq
      } else {
        await this.chatClient.partialUpdateMessage(channelMessage.id, {
          set: { text: message_text },
        });

        await this.channel.sendEvent({
          type: "ai_indicator.clear",
          cid: channelMessage.cid,
          message_id: channelMessage.id,
        });
      }
    }
  }
}
