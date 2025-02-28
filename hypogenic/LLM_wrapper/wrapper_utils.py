def _process_deepseek_messages(messages_list):
    """
    Helper function to process messages for DeepSeek models.
    Moves system message content to the beginning of the first user message.
    """
    if isinstance(messages_list[0], dict):
        # Handle single message list case
        messages = messages_list
        system_contents = []
        new_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_contents.append(msg.get("content", ""))
            else:
                new_messages.append(msg)
        
        if system_contents and new_messages:
            # If there are system messages and other messages, prepend to first user
            for i, msg in enumerate(new_messages):
                if msg.get("role") == "user":
                    new_messages[i]["content"] = "\n".join(system_contents + [msg.get("content", "")])
                    break
            else:
                # If no user message found, create one
                new_messages.insert(0, {"role": "user", "content": "\n".join(system_contents)})
        elif system_contents:
            # Only system messages exist
            new_messages = [{"role": "user", "content": "\n".join(system_contents)}]
        
        return new_messages
    else:
        # Handle nested message lists (for VLLM)
        return [_process_deepseek_messages(msgs) for msgs in messages_list]