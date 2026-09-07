//! Interactive chat command using the high-level Chat API.

use anyhow::{Result, anyhow};
use futures::StreamExt;
use std::io::{self, BufRead, Write};

use kjarni::chat::Chat;

/// Slash commands available in chat mode
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatCommand {
    Quit,
    Clear,
    ClearAll,
    ShowSystem,
    ShowHistory,
    Help,
    Unknown(String),
}

/// Result of processing a slash command
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandResult {
    /// Exit the chat loop
    Exit,
    #[allow(
        dead_code,
        reason = "not referenced yet; kept until the path that needs it lands"
    )]
    /// Continue with normal message processing
    Continue,
    /// Command was handled, prompt for next input
    Handled,
}

pub async fn run(
    model: &str,
    _model_path: Option<&str>, // TODO: Handle custom model paths
    system_prompt: Option<&str>,
    temperature: f32,
    max_tokens: usize,
    gpu: bool,
    quiet: bool,
    draft: Option<&str>,
    draft_tokens: usize,
) -> Result<()> {
    // Fail on an unknown model here, with suggestions, rather than letting the
    // builder report it without context.
    let _ = crate::commands::util::resolve_model(model, Some("decoder"))?;

    // Initialize the Chat instance using the Builder
    let mut builder = Chat::builder(model)
        .temperature(temperature)
        .max_tokens(max_tokens);

    if quiet {
        builder = builder.quiet();
    }

    if gpu {
        // Refuse before the loader starts uploading; an out-of-VRAM upload loses
        // the device and takes the terminal with it.
        if let Some(mt) = kjarni::ModelType::from_cli_name(model) {
            super::util::check_gpu_capacity(mt)?;
        }
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }

    if let Some(draft_name) = draft {
        // Fail here with suggestions rather than inside the builder.
        let _ = crate::commands::util::resolve_model(draft_name, Some("decoder"))?;
        builder = builder.draft(draft_name, draft_tokens);
    }

    if let Some(system) = system_prompt {
        builder = builder.system(system);
    }

    let chat = builder
        .build()
        .await
        .map_err(|e| anyhow!("Failed to initialize chat: {}", e))?;

    // Setup stateful conversation
    let mut convo = chat.conversation();

    // Welcome message
    if !quiet {
        println!(
            "{}",
            format_welcome_message(chat.model_name(), &format!("{:?}", chat.device()))
        );
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle Slash Commands
        if input.starts_with('/') {
            let command = parse_command(input);

            match handle_command(&command, &mut convo, &chat, quiet) {
                CommandResult::Exit => break,
                CommandResult::Handled => continue,
                CommandResult::Continue => {}
            }
        }

        convo.push_user(input);

        // Get the stream
        let stream_result = convo.stream_next().await;

        let mut stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[Error starting stream: {}]", e);
                continue;
            }
        };

        // Collect and print tokens
        let mut full_response = String::new();

        while let Some(token_result) = stream.next().await {
            match token_result {
                Ok(text) => {
                    print!("{}", text);
                    stdout.flush()?;
                    full_response.push_str(&text);
                }
                Err(e) => {
                    eprintln!("\n[Error during generation: {}]", e);
                    break;
                }
            }
        }

        // Add the assistant response to history
        if !full_response.is_empty() {
            convo.push_assistant(full_response.trim());
        }

        println!("\n");
    }

    if !quiet {
        println!("Goodbye!");
    }

    Ok(())
}

/// Parse a slash command from user input
fn parse_command(input: &str) -> ChatCommand {
    match input.to_lowercase().as_str() {
        "/quit" | "/exit" | "/q" => ChatCommand::Quit,
        "/clear" | "/reset" => ChatCommand::Clear,
        "/clearall" => ChatCommand::ClearAll,
        "/system" => ChatCommand::ShowSystem,
        "/history" => ChatCommand::ShowHistory,
        "/help" => ChatCommand::Help,
        _ => ChatCommand::Unknown(input.to_string()),
    }
}

fn handle_command<C>(
    command: &ChatCommand,
    convo: &mut C,
    chat: &Chat,
    quiet: bool,
) -> CommandResult
where
    C: ConversationLike,
{
    match command {
        ChatCommand::Quit => CommandResult::Exit,

        ChatCommand::Clear => {
            convo.clear_history(true);
            if !quiet {
                println!("Conversation history cleared.");
            }
            CommandResult::Handled
        }

        ChatCommand::ClearAll => {
            convo.clear_history(false);
            if !quiet {
                println!("Conversation history cleared (including system prompt).");
            }
            CommandResult::Handled
        }

        ChatCommand::ShowSystem => {
            if let Some(s) = chat.system_prompt() {
                println!("System Prompt: {}", s);
            } else {
                println!("No system prompt set.");
            }
            CommandResult::Handled
        }

        ChatCommand::ShowHistory => {
            let (len, messages) = convo.get_history_info();
            println!("\n--- History ({} messages) ---", len);
            for (role, content) in messages {
                println!("[{}]: {}", role, content);
            }
            println!("--- End ---\n");
            CommandResult::Handled
        }

        ChatCommand::Help => {
            println!("{}", format_help_text());
            CommandResult::Handled
        }

        ChatCommand::Unknown(cmd) => {
            println!(
                "Unknown command: {}. Type /help for available commands.",
                cmd
            );
            CommandResult::Handled
        }
    }
}

pub trait ConversationLike {
    fn clear_history(&mut self, keep_system: bool);
    fn get_history_info(&self) -> (usize, Vec<(String, String)>);
}

impl ConversationLike for kjarni::chat::conversation::ChatConversation<'_> {
    fn clear_history(&mut self, keep_system: bool) {
        self.clear(keep_system);
    }

    fn get_history_info(&self) -> (usize, Vec<(String, String)>) {
        let history = self.history();
        let messages: Vec<(String, String)> = history
            .messages()
            .iter()
            .map(|msg| (msg.role.to_string(), msg.content.clone()))
            .collect();
        (self.len(), messages)
    }
}

fn format_welcome_message(model_name: &str, device: &str) -> String {
    format!(
        "\nKjarni Chat: {}\nDevice: {}\nType '/help' for commands, '/quit' to exit.\n",
        model_name, device
    )
}

fn format_help_text() -> String {
    r#"
Commands:
  /quit, /exit, /q  - Exit chat
  /clear, /reset    - Reset history (keep system prompt)
  /clearall         - Reset history (remove system prompt too)
  /system           - Show system prompt
  /history          - Show conversation history
  /help             - Show this help
"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    struct MockConversation {
        messages: Vec<(String, String)>,
        clear_calls: Vec<bool>,
    }

    impl MockConversation {
        fn new() -> Self {
            Self {
                messages: vec![],
                clear_calls: vec![],
            }
        }

        fn with_messages(messages: Vec<(&str, &str)>) -> Self {
            Self {
                messages: messages
                    .into_iter()
                    .map(|(r, c)| (r.to_string(), c.to_string()))
                    .collect(),
                clear_calls: vec![],
            }
        }
    }

    impl ConversationLike for MockConversation {
        fn clear_history(&mut self, keep_system: bool) {
            self.clear_calls.push(keep_system);
            if keep_system {
                // Keep only system messages
                self.messages.retain(|(role, _)| role == "system");
            } else {
                self.messages.clear();
            }
        }

        fn get_history_info(&self) -> (usize, Vec<(String, String)>) {
            (self.messages.len(), self.messages.clone())
        }
    }

    #[test]
    fn test_parse_command_quit() {
        assert_eq!(parse_command("/quit"), ChatCommand::Quit);
        assert_eq!(parse_command("/exit"), ChatCommand::Quit);
        assert_eq!(parse_command("/q"), ChatCommand::Quit);
    }

    #[test]
    fn test_parse_command_quit_case_insensitive() {
        assert_eq!(parse_command("/QUIT"), ChatCommand::Quit);
        assert_eq!(parse_command("/Quit"), ChatCommand::Quit);
        assert_eq!(parse_command("/EXIT"), ChatCommand::Quit);
        assert_eq!(parse_command("/Q"), ChatCommand::Quit);
    }

    #[test]
    fn test_parse_command_clear() {
        assert_eq!(parse_command("/clear"), ChatCommand::Clear);
        assert_eq!(parse_command("/reset"), ChatCommand::Clear);
    }

    #[test]
    fn test_parse_command_clear_case_insensitive() {
        assert_eq!(parse_command("/CLEAR"), ChatCommand::Clear);
        assert_eq!(parse_command("/Clear"), ChatCommand::Clear);
        assert_eq!(parse_command("/RESET"), ChatCommand::Clear);
    }

    #[test]
    fn test_parse_command_clearall() {
        assert_eq!(parse_command("/clearall"), ChatCommand::ClearAll);
        assert_eq!(parse_command("/CLEARALL"), ChatCommand::ClearAll);
        assert_eq!(parse_command("/ClearAll"), ChatCommand::ClearAll);
    }

    #[test]
    fn test_parse_command_system() {
        assert_eq!(parse_command("/system"), ChatCommand::ShowSystem);
        assert_eq!(parse_command("/SYSTEM"), ChatCommand::ShowSystem);
    }

    #[test]
    fn test_parse_command_history() {
        assert_eq!(parse_command("/history"), ChatCommand::ShowHistory);
        assert_eq!(parse_command("/HISTORY"), ChatCommand::ShowHistory);
    }

    #[test]
    fn test_parse_command_help() {
        assert_eq!(parse_command("/help"), ChatCommand::Help);
        assert_eq!(parse_command("/HELP"), ChatCommand::Help);
    }

    #[test]
    fn test_parse_command_unknown() {
        assert_eq!(
            parse_command("/foo"),
            ChatCommand::Unknown("/foo".to_string())
        );
        assert_eq!(
            parse_command("/unknown"),
            ChatCommand::Unknown("/unknown".to_string())
        );
    }

    #[test]
    fn test_parse_command_unknown_preserves_case() {
        // Unknown commands should preserve the original case in the error
        match parse_command("/FooBar") {
            ChatCommand::Unknown(cmd) => assert_eq!(cmd, "/FooBar"),
            _ => panic!("Expected Unknown variant"),
        }
    }

    #[test]
    fn test_parse_command_with_extra_text() {
        // Commands with extra text are treated as unknown
        assert!(matches!(
            parse_command("/quit now"),
            ChatCommand::Unknown(_)
        ));
        assert!(matches!(parse_command("/help me"), ChatCommand::Unknown(_)));
    }

    #[test]
    fn test_format_welcome_message() {
        let msg = format_welcome_message("llama-3.2-1b", "Cpu");

        assert!(msg.contains("Kjarni Chat"));
        assert!(msg.contains("llama-3.2-1b"));
        assert!(msg.contains("Cpu"));
        assert!(msg.contains("/help"));
        assert!(msg.contains("/quit"));
    }

    #[test]
    fn test_format_welcome_message_different_model() {
        let msg = format_welcome_message("phi3.5-mini", "Wgpu");

        assert!(msg.contains("phi3.5-mini"));
        assert!(msg.contains("Wgpu"));
    }

    #[test]
    fn test_format_help_text_contains_all_commands() {
        let help = format_help_text();

        assert!(help.contains("/quit"));
        assert!(help.contains("/exit"));
        assert!(help.contains("/q"));
        assert!(help.contains("/clear"));
        assert!(help.contains("/reset"));
        assert!(help.contains("/clearall"));
        assert!(help.contains("/system"));
        assert!(help.contains("/history"));
        assert!(help.contains("/help"));
    }

    #[test]
    fn test_format_help_text_contains_descriptions() {
        let help = format_help_text();

        assert!(help.contains("Exit chat"));
        assert!(help.contains("Reset history"));
        assert!(help.contains("system prompt"));
        assert!(help.contains("conversation history"));
    }

    #[test]
    fn test_chat_command_debug() {
        // Ensure Debug is implemented
        let cmd = ChatCommand::Quit;
        let debug_str = format!("{:?}", cmd);
        assert!(debug_str.contains("Quit"));
    }

    #[test]
    fn test_chat_command_clone() {
        let cmd = ChatCommand::Unknown("test".to_string());
        let cloned = cmd.clone();
        assert_eq!(cmd, cloned);
    }

    #[test]
    fn test_chat_command_eq() {
        assert_eq!(ChatCommand::Quit, ChatCommand::Quit);
        assert_eq!(ChatCommand::Clear, ChatCommand::Clear);
        assert_ne!(ChatCommand::Quit, ChatCommand::Clear);

        assert_eq!(
            ChatCommand::Unknown("a".to_string()),
            ChatCommand::Unknown("a".to_string())
        );
        assert_ne!(
            ChatCommand::Unknown("a".to_string()),
            ChatCommand::Unknown("b".to_string())
        );
    }

    #[test]
    fn test_command_result_debug() {
        let result = CommandResult::Exit;
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("Exit"));
    }

    #[test]
    fn test_command_result_clone() {
        let result = CommandResult::Handled;
        let cloned = result.clone();
        assert_eq!(result, cloned);
    }

    #[test]
    fn test_command_result_eq() {
        assert_eq!(CommandResult::Exit, CommandResult::Exit);
        assert_eq!(CommandResult::Continue, CommandResult::Continue);
        assert_eq!(CommandResult::Handled, CommandResult::Handled);
        assert_ne!(CommandResult::Exit, CommandResult::Continue);
        assert_ne!(CommandResult::Exit, CommandResult::Handled);
        assert_ne!(CommandResult::Continue, CommandResult::Handled);
    }

    #[test]
    fn test_mock_conversation_new() {
        let mock = MockConversation::new();
        let (len, messages) = mock.get_history_info();
        assert_eq!(len, 0);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_mock_conversation_with_messages() {
        let mock =
            MockConversation::with_messages(vec![("user", "hello"), ("assistant", "hi there")]);
        let (len, messages) = mock.get_history_info();
        assert_eq!(len, 2);
        assert_eq!(messages[0], ("user".to_string(), "hello".to_string()));
        assert_eq!(
            messages[1],
            ("assistant".to_string(), "hi there".to_string())
        );
    }

    #[test]
    fn test_mock_conversation_clear_keep_system() {
        let mut mock = MockConversation::with_messages(vec![
            ("system", "You are helpful"),
            ("user", "hello"),
            ("assistant", "hi"),
        ]);

        mock.clear_history(true);

        let (len, messages) = mock.get_history_info();
        assert_eq!(len, 1);
        assert_eq!(messages[0].0, "system");
        assert_eq!(mock.clear_calls, vec![true]);
    }

    #[test]
    fn test_mock_conversation_clear_all() {
        let mut mock =
            MockConversation::with_messages(vec![("system", "You are helpful"), ("user", "hello")]);

        mock.clear_history(false);

        let (len, _) = mock.get_history_info();
        assert_eq!(len, 0);
        assert_eq!(mock.clear_calls, vec![false]);
    }

    #[test]
    fn test_parse_and_identify_quit_commands() {
        let quit_inputs = vec!["/quit", "/exit", "/q", "/QUIT", "/Exit", "/Q"];

        for input in quit_inputs {
            let cmd = parse_command(input);
            assert_eq!(cmd, ChatCommand::Quit, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_parse_and_identify_clear_commands() {
        let clear_inputs = vec!["/clear", "/reset", "/CLEAR", "/Reset"];

        for input in clear_inputs {
            let cmd = parse_command(input);
            assert_eq!(cmd, ChatCommand::Clear, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_all_valid_commands_recognized() {
        let commands = vec![
            ("/quit", ChatCommand::Quit),
            ("/exit", ChatCommand::Quit),
            ("/q", ChatCommand::Quit),
            ("/clear", ChatCommand::Clear),
            ("/reset", ChatCommand::Clear),
            ("/clearall", ChatCommand::ClearAll),
            ("/system", ChatCommand::ShowSystem),
            ("/history", ChatCommand::ShowHistory),
            ("/help", ChatCommand::Help),
        ];

        for (input, expected) in commands {
            let result = parse_command(input);
            assert_eq!(result, expected, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_parse_command_empty_after_slash() {
        // Just "/" should be unknown
        let cmd = parse_command("/");
        assert!(matches!(cmd, ChatCommand::Unknown(_)));
    }

    #[test]
    fn test_parse_command_spaces() {
        // Commands with leading/trailing spaces in the unknown part
        let cmd = parse_command("/ help");
        assert!(matches!(cmd, ChatCommand::Unknown(_)));
    }

    #[test]
    fn test_format_welcome_message_special_chars() {
        let msg = format_welcome_message("model-with-special_chars.v1", "Device<Wgpu>");
        assert!(msg.contains("model-with-special_chars.v1"));
        assert!(msg.contains("Device<Wgpu>"));
    }

    #[test]
    fn test_mock_conversation_multiple_clears() {
        let mut mock = MockConversation::with_messages(vec![
            ("system", "sys"),
            ("user", "u1"),
            ("assistant", "a1"),
        ]);

        mock.clear_history(true);
        mock.clear_history(false);

        assert_eq!(mock.clear_calls, vec![true, false]);
        let (len, _) = mock.get_history_info();
        assert_eq!(len, 0);
    }

    #[test]
    fn test_mock_conversation_unicode() {
        let mock =
            MockConversation::with_messages(vec![("user", "こんにちは"), ("assistant", "你好")]);

        let (len, messages) = mock.get_history_info();
        assert_eq!(len, 2);
        assert_eq!(messages[0].1, "こんにちは");
        assert_eq!(messages[1].1, "你好");
    }

    #[test]
    fn test_mock_conversation_empty_messages() {
        let mock = MockConversation::with_messages(vec![("user", ""), ("assistant", "")]);

        let (len, messages) = mock.get_history_info();
        assert_eq!(len, 2);
        assert_eq!(messages[0].1, "");
        assert_eq!(messages[1].1, "");
    }

    #[test]
    fn test_mock_conversation_long_content() {
        let long_content = "a".repeat(10000);
        let mock = MockConversation::with_messages(vec![("user", &long_content)]);

        let (_, messages) = mock.get_history_info();
        assert_eq!(messages[0].1.len(), 10000);
    }
}
