import subprocess
import sys
import os
from time import sleep
import boto3
import botocore
from botocore.client import ClientError
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import io
import json
from opensearchpy import OpenSearch
from opensearchpy import RequestsHttpConnection, OpenSearch, AWSV4SignerAuth

import boto3
import os



#Setup Bedrock client
config = botocore.config.Config(connect_timeout=500, read_timeout=500)
bedrock = boto3.client('bedrock-runtime' , 'us-east-1', config = config)

#s3 client
s3 = boto3.client('s3')

#Setup Opensearch connectionand clinet
host = '14dzfsbbbt70yuz57f23.us-west-2.aoss.amazonaws.com' #use Opensearch Serverless host here
region = 'us-west-2'# set region of you Opensearch severless collection
service = 'aoss'
credentials = boto3.Session().get_credentials() #Use enviroment credentials
auth = AWSV4SignerAuth(credentials, region, service) 

oss_client = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    pool_maxsize = 20
)


st.set_page_config(page_title="Bedrock Architect Assistant", page_icon=":tada", layout="wide")

#Headers
with st.container():
    st.header("Bedrock Solution Architect Assistant")
    st.title("Upload your an image of your architecture")



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
        


def get_embeddings(bedrock, text):
    body_text = json.dumps({"inputText": text})
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType='application/json'

    response = bedrock.invoke_model(body=body_text, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    return embedding

def image_to_description(image_name, file_type):

    #open file and convert to base64
    open_image = Image.open(image_name)
    image_bytes = io.BytesIO()
    open_image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')



    #setup prompt
    system_prompt=f"""
You are an AWS Solutions Architect. You will be provided an image or drawing of an architecture diagram or AWS based workflow. 
Based on the image your goal is to provide A high level overview of the type of workflow/architecture is depicted in the diagram and what the solution shown will provide - only comment on the architecture diagram, nothing else in the image   
The overview should provide information such as the type of application, the design pattern, and the primary components and services etc. in the image. This should be from the perspective of a Solutions Architect that drew the diagram, and should be technical in nature
Dont include bullet points in your response, just a concise overview of the architecture
The architecture should make sense as a valid solution, but dont make assumptions

Example output format, use this example for the response format:
<overview>
(Brief description of what the image is about from the perspective of an AWS solutions Architect)
</description>

If the image provided is not an architecture diagram, respond with "Not Valid" for your response

Provide your thoughts in <thoughts> xml tags
Respond with your high level overview in <overview> xml tags, return "Not Valid" if the image is not an architecture diagram. No other text


"""

    prompt = {
        "anthropic_version":"bedrock-2023-05-31",
        "max_tokens":10000,
        "temperature":0.1,
        "system": system_prompt,
        "messages":[
            {
                "role":"user",
                "content":[
                {
                    "type":"image",
                    "source":{
                        "type":"base64",
                        "media_type":file_type,
                        "data": image_base64
                    }
                }
                ]
            }
        ]
    }

    json_prompt = json.dumps(prompt)

    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")


    response_body = json.loads(response.get('body').read())

    llmOutput=response_body['content'][0]['text']

    print(llmOutput)

    thoughts = parse_xml(llmOutput, "thoughts")
    overview = parse_xml(llmOutput, "overview").strip()


    return overview


def image_to_details(image_name, file_type):

    #open file and convert to base64
    open_image = Image.open(image_name)
    image_bytes = io.BytesIO()
    open_image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')



    #setup prompt
    system_prompt=f"""
You are an AWS Solutions Architect. You will be provided an image of an architecture diagram. 
Based on the image your goal is to provideA detailed description of all the components and AWS services depicted in the diagram, and their relationships to each other, which components are contained within each other, how components are connected to each other and how the data/workflow flows
This should be written from the perspective of the Solutions architect who created the diagram with the goal of documenting how the infrastructure was built
The Details should provide robust but concise information and of the following
     The grouping of each component/service - i.e. AZ 1 contains x,y,z. Public Subnet 1 contains z,y,z etc.
     The responsibility of each component
     The relationship between the groupings and components
     The workflow/data flow of the overall solution

Use the below example as a guide for you output format:
<format_example>
The image depicts a high-level architecture diagram for a web application hosted on AWS (Amazon Web Services). The key components and their relationships are as follows:
	1. Web App: This represents the web application that users will interact with.
	2. Internet: The web application is accessible over the internet.
	3. AWS: The web application is hosted on the AWS cloud platform. Within AWS, the architecture consists of the following components:
	• Availability Zones (AZs):
		○ AZ 1: This availability zone contains the following resources:
			§ Public Subnet: A public subnet that allows inbound internet traffic.
				□ EC2 Web App: One or more EC2 instances running the web application, behind the load balancer.
			§ Private Subnet: A private subnet with no direct internet access.
				□ RDS: An Amazon Relational Database Service (RDS) instance, likely used to store and retrieve data for the web application.
				
		○ AZ 2: This is a second availability zone for high availability and fault tolerance. It contains the following resources:
			§ Public Subnet: A public subnet that allows inbound internet traffic.
				□ EC2 Web App: One or more EC2 instances running the web application.
			§ Private Subnet: A private subnet with no direct internet access.
				□ RDS: An Amazon Relational Database Service (RDS) instance.
	• ALB: An Application Load Balancer (ALB) that distributes incoming traffic across the two availability zones (AZ 1 and AZ 2) for high availability and fault tolerance.
	• Replication: There is data replication between the RDS instances in the two availability zones to ensure data consistency and availability in case of a failure in one availability zone.
	
The workflow is as follows:
	1. Users access the web application over the internet.
	2. The Application Load Balancer (ALB) distributes incoming traffic across the two availability zones (AZ 1 and AZ 2).
	3. In each availability zone, the Elastic Load Balancer (ELB) distributes traffic across the EC2 instances running the web application.
	4. The web application instances retrieve and store data from/to the RDS instances in the private subnets.
	5. Data is replicated between the RDS instances in the two availability zones for high availability and fault tolerance.

</format_example>

Your response will be used to construct a diagram and a Cloud Formation template, so ensure that everything that is grouped together is captured as such in your details
Pay extra attention to accuracy and including everything that will be needed to recreate a diagram, and a fully functioning Cloud Formation Infrastructure as Code template

Think through each step of your thought process, and pay extra attention to providing accurate details.
    
Provide your response in <details> xml tags
"""

    prompt = {
        "anthropic_version":"bedrock-2023-05-31",
        "max_tokens":10000,
        "temperature":0.1,
        "system": system_prompt,
        "messages":[
            {
                "role":"user",
                "content":[
                {
                    "type":"image",
                    "source":{
                        "type":"base64",
                        "media_type":file_type,
                        "data": image_base64
                    }
                }
                ]
            }
        ]
    }

    json_prompt = json.dumps(prompt)

    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")


    response_body = json.loads(response.get('body').read())

    llmOutput=response_body['content'][0]['text']

    print(llmOutput)

    details = parse_xml(llmOutput, "details")

    return details



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




def description_to_template(overview, details):

    #setup prompt
    system_prompt=f"""
You are an AWS Solutions Architect. You will be provided the description and the details of an application/workflow
Using that description and the details provided, generate an accurate and valid AWS Cloud Formation Template YAML file.

Your Template should capture accurately capture the architectural components, their groupings, and relationships
Your template must be able to to successfully deploy the application/workflow described


Think through each step of your thought process, and pay extra attention to providing a valid Infrastructure as code template and configuration.

Provide your thoughts for each step in <thoughts> xml tags (dont mention "<iac_results>" explicitly in your thoughts)
Provide your valid python diagram for the application/workflow in <iac_results> tags, and no other text


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
            }
        ]
    }

    json_prompt = json.dumps(prompt)

    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")


    response_body = json.loads(response.get('body').read())

    llmOutput=response_body['content'][0]['text']

    thoughts = parse_xml(llmOutput, "thoughts")
    iac_results = parse_xml(llmOutput, "iac_results")

    print(thoughts)

    return iac_results


def description_to_template_questions(overview, details):

    #setup prompt
    system_prompt=f"""
You are an AWS Solutions Architect. You will be provided the description and the details of an application/workflow
Using that description and the details provided, you will eventually be asked to generate an accurate and valid AWS Cloud Formation Template YAML file (not yet though).
Based on the provided <overview> and <details>, provide a list of ALL the information that is NOT provided that would be needed in order to fully create a functional and accurate Cloud Formation Template
Only compile questions whose answers are needed to complete the creation of a Cloud Formation template (ex. instance sizes. Networking information, security group info, iam roles, Which region and availability zones etc.). Dont seek to add anything new if it is not nessicary for the template
Only ask questions that you NEED to create a Cloud Formation template
Return your list of concise questions that need to be answered in order to create a valid Cloud Formation Template
Do your best to group relevant question together, by component and order in the workflow


Think through each step of your thought process, making sure to account for all the things you would need

Provide your thoughts for each step in <thoughts> xml tags (dont mention "<questions>" explicitly in your thoughts)
Provide your list of key missing information/question <questions> tags, and no other text


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
            }
        ]
    }

    json_prompt = json.dumps(prompt)

    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")


    response_body = json.loads(response.get('body').read())

    llmOutput=response_body['content'][0]['text']

    thoughts = parse_xml(llmOutput, "thoughts")
    questions = parse_xml(llmOutput, "questions")

    print(thoughts)

    return questions


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



def evaluate_diagram(details, diagram, original_image, resulting_image):

        #open file and convert to base64
    original_open_image = Image.open(original_image)
    image_bytes_1 = io.BytesIO()
    original_open_image.save(image_bytes_1, format='PNG')
    image_bytes_1 = image_bytes_1.getvalue()
    original_image_base64 = base64.b64encode(image_bytes_1).decode('utf-8')


    resulting_open_image = Image.open(resulting_image)
    image_bytes_2 = io.BytesIO()
    resulting_open_image.save(image_bytes_2, format='PNG')
    image_bytes_2 = image_bytes_2.getvalue()
    resulting_image_base64 = base64.b64encode(image_bytes_2).decode('utf-8')


    #setup prompt
    system_prompt=f"""
You are a Cloud Solutions Architect. 
You will be provided an image of a draw.io diagram for an AWS Application architecture diagram that you created (aka "Generated Diagram"). You will also be provided the original diagram image (aka "Original Diagram") and details that you used to create the existing draw.io diagram
Lastly you will be provided the xml code fo the draw.io diagram that created the image (for the Generated Diagram), which you created

Evaluate if the existing image is a good match to the original and maintains all the details that were provided
You should be evaluating the positioning, organization, and components of the diagram you created, and if it looks to be a professional grade quality
If it is not, provide very detailed instructions and prescriptive guidance on what needs to be adjusted, changed, added, or removed and how you would make those changes in great detail
Your suggested changes should relate to the diagram xml code and how to fix that.
Your instructions should NOT reference the original image and should be understandable without the original image. Be as specific as possible when describing how to make the changes (including what xml code should be changed and how)

If the resulting image is well organized and looks good in comparison (they dont need to be exact matches) good in comparison with the original image, and the quality is high, respond with "Good" in <evaluation> xml tags
If the results dont match, response with "Needs Improvement" in <evaluation> xml tags
Provide your detailed instructions in <instructions> xml tags on how to fix the xml diagram provided to better represent the image



etag should resemble something like this: etag="JSBBcipcDYjVQKObEopO" (the etag you create should not contain "Gu_Gu")


Think through each step of your thought process
Pay extra attention the components, positioning, and relationship of the components in the diagram
Provide your thoughts for each step in <your_thoughts> xml tags (dont mention "<evaluation>" or "<instructions>" in your thoughts)
Provide your evaluation ("Good" or "Needs Improvement") in <evaluation> xml tags, with no other text
Provide your detailed instructions in <instructions> xml tags, with no other text

"""
    
    user_prompt=f"""
<original_details>
{details}
</original_details>

<generated_diagram_xml>
{diagram}
</generated_diagram_xml>
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
                        "type": "text",
                        "text": "Image 1 - The Original Diagram"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type":"base64",
                            "media_type":"image/png",
                            "data": original_image_base64
                        },
                    },
                    {
                        "type":"text",
                        "text": "Image 2 - The Generated Diagram"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type":"base64",
                            "media_type":"image/png",
                            "data": resulting_image_base64
                        },
                    },
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
    output_tokens = int(response_body["usage"]["output_tokens"])
    

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
        
            finished_response, status, tokens = diagram_modification(diagram,input,full_response)

            #merge responses
            full_response = full_response + finished_response

            total_tokens = total_tokens+int(tokens)

            print(f"TOTAL TOKENS: {total_tokens}")

            if status =="max_tokens":
                max_exceeded = True
            else:
                max_exceeded = False

            count += 1

        print("I MADE IT OUT OF THE LOOP!")

        full_response = full_response.replace('</drawio_diagram_results>', '')
        full_response = full_response.replace('<drawio_diagram_results>', '')




        
    else:
        print("Eval finished in one go")

    thoughts = parse_xml(llmOutput, "thoughts")
    evaluation = parse_xml(llmOutput, "evaluation")
    instructions = parse_xml(llmOutput, "instructions")

    return thoughts, evaluation, instructions


def diagram_modification(details, diagram, modification_instructions, examples):


    #setup prompt
    system_prompt=f"""
You are an AWS Solutions Architect. 
You will be provided the details and draw.io xml code of a architecture diagram
You will also be provided the instructions that you need to modify the diagram to better represent the details provided
Using that the using the modification instructions, adjust the draw.io xml code to better represent an accurate and valid architecture diagram representing the workflow using draw.io xml format

Your diagram should capture accurately capture the architectural components, the groupings, the relationships, and the positions of the components as provided by the workflow details. 
Use the modification instructions to make the diagram more accurate and higher quality.
Only change code that needs to be adjusted to adhere to the modification instructions. Dont modify anything that doest need modification as per the instructions

Use the provided few shot example diagrams below as a guide for you draw.io format and style:
{examples}


Think through each step of your thought process, and pay extra attention to providing a valid xml code for draw.io that captures the adjustments and modifications provided
Pay extra attention the components, positioning, and relationship of the components in the diagram
Provide your thoughts for each step in <your_thoughts> xml tags (dont mention "modified_diagram_results" in your thoughts)
Provide your draw.io xml code for the application/workflow surrounded by <modified_diagram_results></modified_diagram_results> xml tags, and no other text


"""
    
    user_prompt=f"""
<workflow_details>
{details}
</workflow_details>

<existing_diagram_xml>
{diagram}
</existing_diagram_xml>

<modification_instructions>
{modification_instructions}
</modification_instructions>
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
        unfinished_response = llmOutput.split('<modified_diagram_results>')
        full_response = unfinished_response[1]
        max_tries = 6
        count = 0

        while max_exceeded == True and count < max_tries:
            print("I AM IN THE LOOP!!!!")

            #call "continue" method
        
            finished_response, status = diagram_modification_continuation(overview,details,examples,full_response, thoughts)

            #merge responses
            full_response = full_response + finished_response


            if status =="max_tokens":
                max_exceeded = True
            else:
                max_exceeded = False

            count += 1

        print("I MADE IT OUT OF THE LOOP!")
        print(unfinished_response)

        full_response = full_response.replace('</modified_diagram_results>', '')
        full_response = full_response.replace('<modified_diagram_results>', '')




        
    else:
        full_response = parse_xml(llmOutput, "modified_diagram_results")

    


    return full_response


def diagram_modification_continuation(overview,details, examples, previous_response, thought_process):

    previous_response = previous_response.rstrip()

    #setup prompt
    system_prompt=f"""
You are an AWS Solutions Architect. You have been asked to modify and adjust a draw.io xml diagram depicting the architecture provided to you in <workflow_details> based on the provided <modifcation_instructions>
While generating the adjusted draw.io xml, you hit the maximum output token length.
Please continue where you left off and complete the unfinished draw.io diagram provided in <unfinished_diagram>


Your diagram should capture accurately capture the architectural components, the groupings, the relationships, and the positions of the components as provided by the workflow details. 
Use the modification instructions to make the diagram more accurate and higher quality
Only change code that needs to be adjusted to adhere to the modification instructions. Dont modify anything that doest need modification as per the instructions

Use the provided few shot example diagrams below as a guide for you draw.io format and style:
{examples}


Think through each step of your thought process, and pay extra attention to providing a valid xml code for draw.io that captures the adjustments and modifications provided
Pay extra attention the components, positioning, and relationship of the components in the diagram


Only provide your continuation of the diagram. When put together, your previous diagram and your continuation should make a fully valid and complete draw.io xml diagram


Think through each step of your thought process, and pay extra attention to providing a valid xml code for draw.io that captures the workflow described
Pay extra attention the components, positioning, and relationship of the components in the diagram
Finish the diagram if you can
Provide your valid draw.io diagram starting from the unfinished_diagram. When complete add </modified_diagram_results> xml tag to mark your completion. Include no other text in your response


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

    

    print("MODIFED OUTPUT")
    print(continued_drawio_diagram_results)
    print("MODIFIED------------------------------------------------------------------")

    return continued_drawio_diagram_results, stop_reason


def validate_cf(template):
    cf_client = boto3.client('cloudformation')
    try:
        cf_client.validate_template(TemplateBody=template)
        return True
    except ClientError as e:
        print(e)
        return False




#Get KNN Results
def get_knn_diagrams(client, overview_vectors, filter, ks):

    query = {
    "size": ks,
    "query": {
        "bool": {
            "filter": {
                "term": {
                    "type": filter
                }
            },
            "must": {
                "knn": {
                    "vectors": {
                        "vector": overview_vectors,
                        "k": ks
                    }
                }
            }
        }
    },
    "_source": False,
    "fields": ["content"],
}


    response = client.search(
        body=query,
        index='diagram',
    )

    score = []


    similaritysearchResponse = ""
    count = 1
    for i in response["hits"]["hits"]:
        content = str(i["fields"]["content"][0])
        print(content)

        score.append(float(i["_score"]))
        print(f"Score: {i['_score']}")

        new_line = "\n"
        prefix = f"<drawio_few_shot_example_{count}>"
        suffix = f"</drawio_few_shot_example_{count}>"

        similaritysearchResponse =  similaritysearchResponse + prefix + new_line + content + new_line + suffix + new_line + new_line
        count = count + 1
    
    print("----------------------Similarity Search Results-----------------------")
    print(similaritysearchResponse)
    print("---------------------END Similarity Search Results--------------------")

    return similaritysearchResponse, score

#Save the diagram to a file (and removes the first line if its blank)
def save_xml_to_file(xml_content, file_path):
    # Split the XML content into lines
    lines = xml_content.splitlines()
    
    # Check if the first line is empty
    if lines and not lines[0].strip():
        # Remove the first line if it is empty
        lines = lines[1:]
    
    # Join the lines back into a single string
    cleaned_xml_content = "\n".join(lines)
    
    # Save the cleaned XML content to the specified file path
    with open(file_path, 'w') as file:
        file.write(cleaned_xml_content)
    
    return file_path


#Export diagram to PNG and save locally
def export_xml_to_image(xml_file_path, output_image_path, drawio_path):
    # Command to export XML to image using draw.io desktop application
    try:
        command = [
            drawio_path,
            '--export',
            '--format', 'png',
            '--output', output_image_path,
            xml_file_path
        ]
    
    # Run the command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        st.error(f"Error: {e}")

    return output_image_path



#Streamlit workflow
uploaded_file = st.file_uploader("Choose an Image")

cf=st.button("Generate Cloud Formation Template")
if cf:
    iac_type = "Cloud Formation"
    iac_output_type = "YAML"
    file_name = uploaded_file.name
    file_type = uploaded_file.type

    st.write(file_name)
    st.write(file_type)

    with st.expander("Image"):
       st.image(uploaded_file)

    
    overview = image_to_description(uploaded_file, file_type)

    if overview == "Not Valid":
        st.write(f"Image is {overview}")
        sys.exit()
    else:
        with st.expander("Generated Overview"):
            st.write(overview)

    overview_embeddings = get_embeddings(bedrock, overview)
    few_shot_examples, score= get_knn_diagrams(oss_client, overview_embeddings, "draw", 2)


    with st.expander("Identified Few Shots"):
        st.write(few_shot_examples)
       
    details = image_to_details(uploaded_file, file_type)

    #Color code confidence by KNN similarity score
    if score[0] > 0.9:
        st.success("Similar Architectures Found")
    elif score[0] < 0.9 and score[0] >0.8:
        st.warning("Some similar assets found - But not exact")
    else:
        st.error("No similar Architectures found - creating from scratch (results may vary)")

    with st.expander("Similarity Score"):
        for i in score:
            st.write(f"Score: {i}")


    with st.expander("Generated Details"):
        st.write(details)


    with st.expander("Generated Diagram"):
        diagram = description_to_diagram(overview, details, few_shot_examples)
        st.code(diagram, language="xml", line_numbers=True)



    #Save diagram to file
    diagram_file_path = save_xml_to_file(diagram, "diagram.xml")
    st.write(f"Diagram saved as {diagram_file_path}")

    #Export diagram to png
    drawio_path = "C:\\Program Files\\draw.io\\draw.io.exe"
    diagram_file_full_path = os.path.join("C:\\Users\\dbavaro\\Documents\\Blogs-and-Projects\\GenAI\\Demos\\architecture-to-iac", diagram_file_path)
    diagram_image_path = export_xml_to_image(diagram_file_full_path, "diagram.png", drawio_path)


    with st.expander("Diagram Image"):
        try:
            st.image(diagram_image_path)
        except:
            st.write("Image could not be displayed")

    
    with st.expander("Diagram Evaluation"):
        thoughts, evaluation, instructions = evaluate_diagram(details, diagram, uploaded_file, diagram_image_path)
        st.write(f"Thoughts: {thoughts}")
        st.write(f"Evaluation: {evaluation}")
        st.write(f"Instructions: {instructions}")

    with st.expander("Adjusted Diagram"):
        if evaluation == "Good":
            st.success("Original Diagram is Good")
        else:
            modified_diagram = diagram_modification(details,diagram,instructions,few_shot_examples)
            st.write("Original Diagram Needs Improvement")
            st.code(modified_diagram, language="xml", line_numbers=True)



    #Switch to Cloud Formation focus
    questions = description_to_template_questions(overview, details)

    with st.expander("Unresolved Questions for IaC"):
        st.write(questions)

    valid_template = False
    attempts = 0
    max_attempts = 3
    #Validate CloudFormation template
    while not valid_template and attempts < max_attempts:
        template = description_to_template(overview, details)
        valid_template = validate_cf(template)
        if not valid_template:
            st.write(f"Template is invalid. Trying again.")
            attempts += 1  # Increment attempts counter

    if not valid_template:
        st.error("A valid template could not be produced. Try again later.")
    else:
        with st.expander("Validated CloudFormation Template"):
            st.code(template, language="yaml", line_numbers=True)

    





    








    




