1. 调整到 case 为 `long_press`, `swipe`
android_segementation_prompt1.log

tool_system_prompt = (
        "\\n\\n# Next Action Selection\\n\\n"
        "Based on the user query and task progress, you must select the single best action to perform next.\\n\\n"
        "## Available Actions:\\n"
        # 这里我们从您原来的prompt中提取了动作列表，作为上下文
        "* `click`: Click the point on the screen.\\n"
        "* `long_press`: Press the point on the screen.\\n"
        "* `swipe`: Swipe from one point to another.\\n"
        "* `type`: Input text into the activated input box.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait for the change to happen.\\n"
        
        "## Output Format\\n"
        "Return **only** the name of the action you selected from the list (e.g., `long_press`, `swipe`).\\n"
        "**Do not** return JSON, XML, or any other text."
)

2. 点明 screenshot
tool_system_prompt = (
        "\\n\\n# Next Action Selection\\n\\n"
        "The image is a screenshot of mobile phone. Based on the screenshot, user query and task progress, you must select the single best action to perform next.\\n\\n"
        "## Available Actions:\\n"
        # 这里我们从您原来的prompt中提取了动作列表，作为上下文
        "* `click`: Click the point on the screen.\\n"
        "* `long_press`: Press the point on the screen.\\n"
        "* `swipe`: Swipe from one point to another.\\n"
        "* `type`: Input text into the activated input box.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait for the change to happen.\\n"
        
        "## Output Format\\n"
        "Return **only** the name of the action you selected from the list (e.g., `click`).\\n"
        "**Do not** return JSON, XML, or any other text."
)

3. 把 click 置后 
tool_system_prompt = (
        "\\n\\n# Next Action Selection\\n\\n"
        "The image is a screenshot of mobile phone. Based on the screenshot, user query and task progress, you must select the single best action to perform next.\\n\\n"
        "## Available Actions:\\n"
        # 这里我们从您原来的prompt中提取了动作列表，作为上下文
        "* `long_press`: Press the point on the screen.\\n"
        "* `swipe`: Swipe from one point to another.\\n"
        "* `type`: Input text into the activated input box.\\n"
        "* `click`: Click the point on the screen.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait for the change to happen.\\n"
        
        "## Output Format\\n"
        "Return **only** the name of the action you selected from the list (e.g., `click`).\\n"
        "**Do not** return JSON, XML, or any other text."
)

4. click 不放例子中
tool_system_prompt = (
        "\\n\\n# Next Action Selection\\n\\n"
        "The image is a screenshot of mobile phone. Based on the screenshot, user query and task progress, you must select the single best action to perform next.\\n\\n"
        "## Available Actions:\\n"
        # 这里我们从您原来的prompt中提取了动作列表，作为上下文
        "* `click`: Click the point on the screen.\\n"
        "* `long_press`: Press the point on the screen.\\n"
        "* `swipe`: Swipe from one point to another.\\n"
        "* `type`: Input text into the activated input box.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait for the change to happen.\\n"
        
        "## Output Format\\n"
        "Return **only** the name of the action you selected from the list (e.g., `long_press`).\\n"
        "**Do not** return JSON, XML, or any other text."
)


5. type 与 click的顺序前置
tool_system_prompt = (
        "\\n\\n# Next Action Selection\\n\\n"
        "The image is a screenshot of mobile phone. Based on the screenshot, user query and task progress, you must select the single best action to perform next.\\n\\n"
        "## Available Actions:\\n"
        # 这里我们从您原来的prompt中提取了动作列表，作为上下文
        "* `type`: Input text into the activated input box.\\n"
        "* `click`: Click the point on the screen.\\n"
        "* `long_press`: Press the point on the screen.\\n"
        "* `swipe`: Swipe from one point to another.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait for the change to happen.\\n"
        
        "## Output Format\\n"
        "Return **only** the name of the action you selected from the list (e.g., `long_press`).\\n"
        "**Do not** return JSON, XML, or any other text."
)

-- 消融实验记录：
        pope的result 怎么办？ [尝试看是否存在调整的余地]
                    

6. 
tool_system_prompt = (
        "\\n\\n# Next Action Selection\\n\\n"
        "The image is a screenshot of mobile phone. Based on the screenshot, user query and task progress, you must select the single best action to perform next.\\n\\n"
        "## Available Actions:\\n"
        # 这里我们从您原来的prompt中提取了动作列表，作为上下文
        "* `click`: Click the point on the screen.\\n"
        "* `long_press`: Press the point on the screen.\\n"
        "* `swipe`: Swipe from one point to another.\\n"
        "* `type`: Input text into the activated input box.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait for the change to happen.\\n"
        
        "## Output Format\\n"
        "Return **only** the name of the action you selected from the list (e.g., `type`).\\n"
        "**Do not** return JSON, XML, or any other text."
)



7. 
tool_system_prompt = (
        "\\n\\n# Next Action Selection\\n\\n"
        "The image is a screenshot of mobile phone. Based on the screenshot, user query and task progress, you must select the single best action to perform next.\\n\\n"
        "## Available Actions:\\n"
        # 这里我们从您原来的prompt中提取了动作列表，作为上下文
        "* `click`: Click the point on the screen.\\n"
        "* `type`: Input text into the activated input box.\\n"
        "* `long_press`: Press the point on the screen.\\n"
        "* `swipe`: Swipe from one point to another.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait for the change to happen.\\n"
        
        "## Output Format\\n"
        "Return **only** the name of the action you selected from the list based on the screenshot, user query and task progress(e.g., `wait`).\\n"
        "**Do not** return JSON, XML, or any other text."
)



8.
tool_system_prompt = (
        "\\n\\n# Next Action Selection\\n\\n"
        "The image is a screenshot of mobile phone. Based on the screenshot, user query and task progress, you must select the single best action to perform next.\\n\\n"
        "## Available Actions:\\n"
        # 这里我们从您原来的prompt中提取了动作列表，作为上下文
        "* `click`: Click the point on the screen.\\n"
        "* `long_press`: Press the point on the screen.\\n"
        "* `swipe`: Swipe from one point to another.\\n"
        "* `type`: Input text into the activated input box.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait for the change to happen.\\n"
        
        "## Output Format\\n"
        "Return **only** the name of the action you selected from the list based on the screenshot, user query and task progress(e.g., `type`).\\n"
        "**Do not** return JSON, XML, or any other text."
)


9. 
tool_system_prompt = (
        "\\n\\n# Next Action Selection\\n\\n"
        "The image is a screenshot of mobile phone. Based on the screenshot, user query and task progress, you must select the single best action to perform next.\\n\\n"
        "## Available Actions:\\n"
        # 这里我们从您原来的prompt中提取了动作列表，作为上下文
        "* `click`: Click the point on the screen.\\n"
        "* `type`: Input text into the activated input box.\\n"
        "* `long_press`: Press the point on the screen.\\n"
        "* `swipe`: Swipe from one point to another.\\n"
        "* `system_button`: Press the system button.\\n"
        "* `open`: Open an app on the device.\\n"
        "* `wait`: Wait for the change to happen.\\n"
        
        "## Output Format\\n"
        "Return **only** the name of the action you selected from the list based on the screenshot, user query and task progress(e.g., `wait`, `type`).\\n"
        "**Do not** return JSON, XML, or any other text."
)