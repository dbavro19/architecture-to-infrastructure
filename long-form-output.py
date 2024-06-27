def description_to_diagram(overview, details, examples):


    #setup prompt
    system_prompt=f"""
You are an AWS Solutions Architect. You will be provided the description and the details of an application/workflow
Using that description and the details provided, generate an accurate and valid architecture diagram representing the workflow using draw.io xml format

Your diagram should capture accurately capture the architectural components, the groupings, the relationships, and the positions of the components as provided by the workflow details. 
Connect these components together using the workflow details provided
Build the components and workflow EXACTLY as they are described in the <workflow_details>, do not add anything not mentioned. Make no assumptions 
The diagram must be complete, accurate, and in valid draw.io xml format

Use the provided few shot example diagrams below as a guide for you draw.io format and style:
{examples}


etag should resemble something like this: etag="JSBBcipcDYjVQKObEopO" (the etag you create should not contain "Gu_Gu")


Think through each step of your thought process, and pay extra attention to providing a valid xml code for draw.io that captures the workflow described
Pay extra attention the components, positioning, and relationship of the components in the diagram
Provide your thoughts for each step in <your_thoughts> xml tags (dont mention "drawio_diagram_results" in your thoughts)
Provide your draw.io xml code for the application/workflow surrounded by <drawio_diagram_results></drawio_diagram_results> xml tags, and no other text


"""
    
    user_prompt=f"""
<workflow_description>
{overview}
</workflow_description>

<workflow_details>
{details}
</workflow_details>
"""

    prompt = {
        "anthropic_version":"bedrock-2023-05-31",
        "max_tokens":30000,
        "temperature":0,
        "system": system_prompt,
        "messages":[
            {
                "role":"user",
                "content":[
                    {  
                        "type":"text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    }

    json_prompt = json.dumps(prompt)

    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")


    response_body = json.loads(response.get('body').read())

    llmOutput=response_body['content'][0]['text']

    stop_reason = response_body["stop_reason"]

    thoughts = parse_xml(llmOutput, "your_thoughts")

    print("LLMOUTPUT--------------------------------------------------------------------")
    print(llmOutput)
    print("LLMOUTPUT--------------------------------------------------------------------")

    print("STOP REASON_________________________________________________________________")
    print(stop_reason)
    print("STOP REASON_________________________________________________________________")

    if stop_reason == "max_tokens":
        print("THIS HIT THE Output LIMIT")
        max_exceeded=True
        unfinished_response = llmOutput.split('<drawio_diagram_results>')
        full_response = unfinished_response[1]
        max_tries = 6
        count = 0

        while max_exceeded == True and count < max_tries:
            print("I AM IN THE LOOP!!!!")

            #call "continue" method
        
            finished_response, status = diagram_continuation(overview,details,examples,full_response, thoughts)

            #merge responses
            full_response = full_response + finished_response


            if status =="max_tokens":
                max_exceeded = True
            else:
                max_exceeded = False

            count += 1

        print("I MADE IT OUT OF THE LOOP!")
        print(unfinished_response)

        full_response = full_response.replace('</drawio_diagram_results>', '')
        full_response = full_response.replace('<drawio_diagram_results>', '')




        
    else:
        full_response = parse_xml(llmOutput, "drawio_diagram_results")

    


    return full_response


def diagram_continuation(overview,details, examples, previous_response, thought_process):

    previous_response = previous_response.rstrip()

    #setup prompt
    system_prompt=f"""
You are an AWS Solutions Architect. You have been asked to create a draw.io xml diagram depicting the architecture provided to you in <workflow_description> and <workflow_details> and your previous <thought_process>
While generating the draw.io xml, you hit the maximum output token length.
Please continue where you left off and complete the unfinished draw.io diagram provided in <unfinished_diagram>


Your diagram should capture accurately capture the architectural components, the groupings, the relationships, and the positions of the components as provided by the workflow details.
Connect these components together using the workflow details provided
Build the components and workflow EXACTLY as they are described in the <workflow_details>, do not add anything not mentioned. Make no assumptions

Use the provided few shot example diagrams below as a guide for you draw.io format and style:
{examples}


Only provide your continuation of the diagram. When put together, your previous diagram and your continuation should make a fully valid and complete draw.io xml diagram


Think through each step of your thought process, and pay extra attention to providing a valid xml code for draw.io that captures the workflow described
Pay extra attention the components, positioning, and relationship of the components in the diagram
Finish the diagram if you can
Provide your valid draw.io diagram starting from the unfinished_diagram. When complete add </drawio_diagram_results> xml tag to mark your completion. Include no other text in your response


"""
    
    user_prompt=f"""
<workflow_description>
{overview}
</workflow_description>

<workflow_details>
{details}
</workflow_details>

<thought_process>
{thought_process}
</thought_process>
"""

    prompt = {
        "anthropic_version":"bedrock-2023-05-31",
        "max_tokens":10000,
        "temperature":0,
        "system": system_prompt,
        "messages":[
            {
                "role":"user",
                "content":[
                    {  
                        "type":"text",
                        "text": user_prompt
                    }
                ]
            },
            {
                "role" :"assistant",
                "content" : [
                    {
                        "type" : "text",
                        "text" : previous_response
                    }
                ]
            },
        ]
    }

    json_prompt = json.dumps(prompt)

    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")


    response_body = json.loads(response.get('body').read())

    llmOutput=response_body['content'][0]['text']

    print("MODIFIED------------------------------------------------------------------")
    stop_reason = response_body["stop_reason"]
    print(stop_reason)
    print(llmOutput)

    if stop_reason == "max_tokens":
        #logic
        continued_drawio_diagram_results = llmOutput
    else:
        continued_drawio_diagram_results = llmOutput


    thoughts = parse_xml(llmOutput, "thoughts")
    

    print("MODIFED OUTPUT")
    print(continued_drawio_diagram_results)
    print("MODIFIED------------------------------------------------------------------")

    return continued_drawio_diagram_results, stop_reason


def parse_xml(xml, tag):
  start_tag = f"<{tag}>"
  end_tag = f"</{tag}>"
  
  start_index = xml.find(start_tag)
  if start_index == -1:
    return ""

  end_index = xml.find(end_tag)
  if end_index == -1:
    return ""

  value = xml[start_index+len(start_tag):end_index]
  return value