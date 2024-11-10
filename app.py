import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from PIL import Image
import base64
from io import BytesIO
from typing import Literal
from dataclasses import dataclass
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import math
from utils import sim
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

def connect_coppeliaSim():
    try:
        port=19999
        sim.simxFinish(-1) 
        clientID=sim.simxStart('127.0.0.1',port,True,True,2000,5)
        # sim.simxStopSimulation(clientID, )
        if clientID == 0: 
            print("conectado a", port)
            return clientID
        else:
            print("No hay conexion")
            return None
    except:
        return None

def get_image(label_vision_sensor):
    _, vision_sensor =sim.simxGetObjectHandle(st.session_state.clientID, label_vision_sensor, sim.simx_opmode_blocking)  # Vision sensor
    returnCode, resolution, image = sim.simxGetVisionSensorImage(st.session_state.clientID, vision_sensor, 0, sim.simx_opmode_blocking)

    imagen_array = np.array(image, dtype=np.uint8)
    imagen_array = imagen_array.reshape((256, 256, 3))
    imagen_array = np.flipud(np.fliplr(imagen_array))
    imagen = Image.fromarray(imagen_array, mode='RGB')

    imagen.save(f"static/{label_vision_sensor}.png")

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

def initialize_session():
    if "history" not in st.session_state:        
        st.session_state['history'] = []
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'clientID' not in st.session_state:
        st.session_state.clientID = None
    if 'number_objectParent' not in st.session_state:
        st.session_state.number_objectParent = None

    if 'agent' not in st.session_state:

        # Tools to agent_take_objects
        get_postion_final_object_tool = FunctionTool.from_defaults(fn=get_postion_final_object)
        calculate_distance_r_tool = FunctionTool.from_defaults(fn=calculate_distance_r)
        calculate_beta_angle_tool = FunctionTool.from_defaults(fn=calculate_beta_angle)
        calculate_angle_q1_tool = FunctionTool.from_defaults(fn=calculate_angle_q1)
        calculate_angle_q2_tool = FunctionTool.from_defaults(fn=calculate_angle_q2)
        calculate_distance_q3_tool = FunctionTool.from_defaults(fn=calculate_distance_q3)
        set_postion_q1_tool = FunctionTool.from_defaults(fn=set_postion_q1)
        set_postion_q2_tool = FunctionTool.from_defaults(fn=set_postion_q2)
        set_postion_q3_tool = FunctionTool.from_defaults(fn=set_postion_q3)
        get_distance_prooximitySensor_tool = FunctionTool.from_defaults(fn=get_distance_prooximitySensor)
        take_object_tool = FunctionTool.from_defaults(fn=take_object)

        # prompt_context = """
        # Tu objetivo principal es mover un robot SCARA para tomar un objeto especifico, realizando calculos de cinematica inversa y conexiones con CoppeliaSim para mover el robot.
        # Para calcular los angulos de las juntas q1 y q2 y la distancia q3.
        # Debes seguir los siguientes pasos para cumplir tu objetivo:
        # 1. Clasifica el objeto especifico solicitado por el usuario usando alguno de los siguientes nombres: ["Blue_cube", "Red_cube", "Green_cube", "Bowl"], retorna el elemento de la lista que mas se asemeje como el nombre del objeto especifico.
        # 2. Obten la posicion del objeto especifico que tomara el efector final, para ello ejecuta la funcion "get_postion_final_object" y envia como parametro de entrada el nombre del objeto especifico retornado del paso anterior, esto retorna una lista con su posicion: [px, py, pz] esto será util para el calculo.
        # 3. Calcula la distancia "r" entre la base del robot ubicada en (0, 0) y los puntos (px, py) obtenidos del paso anterior, para ello usa la funcion "calculate_distance_r", esto retorna el valor en formato flotante.
        # 4. Calcula el valor del angulo "beta" para ello utiliza las dimensiones del robot (l1 = 0.315 - 0.05, l2 = 0.467003, l3 = 0.400499) y usa el valor de "r" calculado, para ello usa la funcion "calculate_beta_angle", esto retornara el angulo en formato flotante.
        # 5. Usando el angulo "beta" y los puntos (px, py) calcula el angulo "q1" para ello usa la funcion "calculate_angle_q1_tool", esto retornara el valor de angulo "q1" en flotante.
        # 6. Ahora, usa "beta", "q1", "r" y "l3" para calcular el angulo "q2", usa la funcion "calculate_angle_q2_tool", esto retornara el valor del angulo "q2" en formato flotante.
        # 7. Para calular la distancia "q3" ejecuta la funcion "calculate_distance_q3_tool", ten en cuenta el punto "pz" y "l1", esto retornara el valor de distancia de "q3" en flotante.
        # 8. Ten muy presentes los valores de "q1", "q2" y "q3".
        # 9. Ahora envia el angulo de posicion "q1" al robot, para ello usa la funcion "set_postion_q1_tool" con el parametro de entrada "q1", esto retornara un valor flotante, debes validar que el valor retornado sea igual al valor de entrada "q1".
        # 10. Ahora envia el angulo de posicion "q2" al robot, para ello usa la funcion "set_postion_q2_tool" con el parametro de entrada "q2", esto retornara un valor flotante, debes validar que el valor retornado sea igual al valor de entrada "q2".
        # 11. Ahora envia la distancia de posicion "q3" al robot, para ello usa la funcion "set_postion_q3_tool" con el parametro de entrada "q3", esto retornara un valor flotante, debes validar que el valor retornado sea igual al valor de entrada "q3".
        # 12. Ejecuta la funcion "get_distance_prooximitySensor_tool" para evaluar si es posible tomar con el efector final el objeto, esta retornara un valor entero que hace referencia al ID
        #  del objeto a tomars si es que se puede tomar, si no retornara None.
        # 13. Dependiendo el valor obtenido del paso anterior, si es un valor entero ejecuta la funcion "take_object_tool" para tomar el objeto.
        # 14. Ahora envia la distancia de posicion 0.30 (float) al robot, para ello usa la funcion "set_postion_q3_tool" con el parametro de entrada "0.30 (float)", esto retornara un valor flotante, debes validar que el valor retornado sea igual al valor de entrada "0.30 (float)".
        # """

        prompt_context = """
        Your main objective is to move a SCARA robot to pick up a specific object, performing inverse kinematics calculations and connections with CoppeliaSim to move the robot.
        To calculate the joint angles q1 and q2 and the distance q3.
        You must follow the following steps to accomplish your objective:
        1. Classify the specific object requested by the user using one of the following names: ["Blue_cube", "Red_cube", "Green_cube", "Bowl"], return the list item that most closely resembles the name of the specific object.
        2. Obtain the position of the specific object that the end effector will pick up, to do this run the "get_postion_final_object" function and send as input parameter the name of the specific object returned from the previous step, this returns a list with its position: [px, py, pz] this will be useful for the calculation.
        3. Calculate the distance "r" between the robot base located at (0, 0) and the points (px, py) obtained from the previous step, for this use the "calculate_distance_r" function, this returns the value in float format.
        4. Calculate the value of the angle "beta" for this use the dimensions of the robot (l1 = 0.315 - 0.05, l2 = 0.467003, l3 = 0.400499) and use the value of "r" calculated, for this use the "calculate_beta_angle" function, this will return the angle in float format.
        5. Using the angle "beta" and the points (px, py) calculate the angle "q1" for this use the "calculate_angle_q1_tool" function, this will return the value of angle "q1" in float format.
        6. Now, use "beta", "q1", "r" and "l3" to calculate the angle "q2", use the "calculate_angle_q2_tool" function, this will return the value of the angle "q2" in float format.
        7. To calculate the distance "q3" run the "calculate_distance_q3_tool" function, take into account the point "pz" and "l1", this will return the value of the distance of "q3" in float format.
        8. Keep in mind the values ​​of "q1", "q2" and "q3".
        9. Now send the position angle "q1" to the robot, to do this use the "set_postion_q1_tool" function with the input parameter "q1", this will return a float value, you must validate that the value returned is equal to the input value "q1".
        10. Now send the position angle "q2" to the robot, for this use the function "set_postion_q2_tool" with the input parameter "q2", this will return a float value, you must validate that the returned value is equal to the input value "q2".
        11. Now send the position distance "q3" to the robot, for this use the function "set_postion_q3_tool" with the input parameter "q3", this will return a float value, you must validate that the returned value is equal to the input value "q3".
        12. Execute the function "get_distance_prooximitySensor_tool" to evaluate if it is possible to take the object with the end effector, this will return an integer value that refers to the ID
        of the object to be taken if it can be taken, if not it will return None.
        13. Depending on the value obtained from the previous step, if it is an integer value, execute the "take_object_tool" function to take the object.
        14. Now send the position distance 0.30 (float) to the robot, for this use the "set_postion_q3_tool" function with the input parameter "0.30 (float)", this will return a float value, you must validate that the returned value is equal to the input value "0.30 (float)".
        """

        llm = NVIDIA("meta/llama-3.1-70b-instruct", api_key=st.secrets["Nvidia_key"])

        agent_take_objects = ReActAgent.from_tools([get_postion_final_object_tool, calculate_distance_r_tool, calculate_beta_angle_tool,
                                                        calculate_angle_q1_tool, calculate_angle_q2_tool, calculate_distance_q3_tool,
                                                        set_postion_q1_tool, set_postion_q2_tool, set_postion_q3_tool, get_distance_prooximitySensor_tool,
                                                        take_object_tool], 
                                                       llm=llm, context=prompt_context, verbose=True, max_iterations=40)
        

        # Tools to agent_drop_objects
        drop_object_tool = FunctionTool.from_defaults(fn=drop_object)

        # prompt_context = """
        # Tu objetivo principal es mover un robot SCARA para soltar un objeto especifico, realizando calculos de cinematica inversa y conexiones con CoppeliaSim para mover el robot.
        # Para calcular los angulos de las juntas q1 y q2 y la distancia q3.
        # Debes seguir los siguientes pasos para cumplir tu objetivo:
        # 1. Clasifica el lugar especifico solicitado por el usuario para soltar el objeto usando alguno de los siguientes nombres: ["Blue_cube", "Red_cube", "Green_cube", "Bowl"], retorna el elemento de la lista que mas se asemeje al lugar especifico.
        # 2. Obten la posicion del nombres del lugar especifico clasificado donde el efector final soltar el objeto, para ello ejecuta la funcion "get_postion_final_object" y envia como parametro de entrada el nombre del lugar especifico retornado del paso anterior, esto retorna una lista con su posicion: [px, py, pz] esto será util para el calculo.
        # 3. Calcula la distancia "r" entre la base del robot ubicada en (0, 0) y los puntos (px, py) obtenidos del paso anterior, para ello usa la funcion "calculate_distance_r", esto retorna el valor en formato flotante.
        # 4. Calcula el valor del angulo "beta" para ello utiliza las dimensiones del robot (l1 = 0.315 - 0.05, l2 = 0.467003, l3 = 0.400499) y usa el valor de "r" calculado, para ello usa la funcion "calculate_beta_angle", esto retornara el angulo en formato flotante.
        # 5. Usando el angulo "beta" y los puntos (px, py) calcula el angulo "q1" para ello usa la funcion "calculate_angle_q1_tool", esto retornara el valor de angulo "q1" en flotante.
        # 6. Ahora, usa "beta", "q1", "r" y "l3" para calcular el angulo "q2", usa la funcion "calculate_angle_q2_tool", esto retornara el valor del angulo "q2" en formato flotante.
        # 7. Ten muy presentes los valores de "q1" y "q2".
        # 8. Ahora envia el angulo de posicion "q1" al robot, para ello usa la funcion "set_postion_q1_tool" con el parametro de entrada "q1", esto retornara un valor flotante, debes validar que el valor retornado sea igual al valor de entrada "q1".
        # 9. Ahora envia el angulo de posicion "q2" al robot, para ello usa la funcion "set_postion_q2_tool" con el parametro de entrada "q2", esto retornara un valor flotante, debes validar que el valor retornado sea igual al valor de entrada "q2".
        # 10. Ejecuta la funcion "drop_object_tool" para soltar el objeto en el lugar especifico.
        # """

        prompt_context = """
        Your main objective is to move a SCARA robot to drop a specific object, performing inverse kinematics calculations and connections with CoppeliaSim to move the robot.
        To calculate the angles of the joints q1 and q2 and the distance q3.
        You must follow the following steps to achieve your objective:
        1. Classify the specific place requested by the user to drop the object using one of the following names: ["Blue_cube", "Red_cube", "Green_cube", "Bowl"], return the element of the list that most resembles the specific place.
        2. Obtain the position of the names of the specific classified place where the final effector will drop the object, to do this execute the function "get_postion_final_object" and send as input parameter the name of the specific place returned from the previous step, this returns a list with its position: [px, py, pz] this will be useful for the calculation.
        3. Calculate the distance "r" between the robot base located at (0, 0) and the points (px, py) obtained from the previous step, for this use the "calculate_distance_r" function, this returns the value in float format.
        4. Calculate the value of the angle "beta" for this use the dimensions of the robot (l1 = 0.315 - 0.05, l2 = 0.467003, l3 = 0.400499) and use the value of "r" calculated, for this use the "calculate_beta_angle" function, this will return the angle in float format.
        5. Using the angle "beta" and the points (px, py) calculate the angle "q1" for this use the "calculate_angle_q1_tool" function, this will return the value of angle "q1" in float format.
        6. Now, use "beta", "q1", "r" and "l3" to calculate the angle "q2", use the "calculate_angle_q2_tool" function, this will return the value of the angle "q2" in float format.
        7. Keep in mind the values ​​of "q1" and "q2".
        8. Now send the position angle "q1" to the robot, for this use the "set_postion_q1_tool" function with the input parameter "q1", this will return a float value, you must validate that the returned value is equal to the input value "q1".
        9. Now send the position angle "q2" to the robot, for this use the "set_postion_q2_tool" function with the input parameter "q2", this will return a float value, you must validate that the returned value is equal to the input value "q2".
        10. Run the "drop_object_tool" function to drop the object at the specific location.
        """

        agent_drop_objects = ReActAgent.from_tools([get_postion_final_object_tool, calculate_distance_r_tool, calculate_beta_angle_tool,
                                                        calculate_angle_q1_tool, calculate_angle_q2_tool, calculate_distance_q3_tool,
                                                        set_postion_q1_tool, set_postion_q2_tool, set_postion_q3_tool, drop_object_tool], 
                                                       llm=llm, context=prompt_context, verbose=True, max_iterations=30)


        agents = {"agent_take_objects": agent_take_objects, "agent_drop_objects": agent_drop_objects}

        query_engine_tools = []
        for agent_name, agent in agents.items():
            tool = QueryEngineTool(
                query_engine=agent,
                metadata=ToolMetadata(
                    name = f"tool_{agent_name}",
                    description = f"This tool uses the {agent_name} to answer questions",
                ),
            )
            query_engine_tools.append(tool)


        # system_prompt = """
        # Eres un agente encargado de guiar a los usuarios en una interfaz que muestra 3 vistas de una escena de un robot SCARA y varios objetos de una simulacion de CoopeliaSim, los usuarios pueden interacturar con el simulador 
        # a traves de ti, para ello deben definir si quieren recoger un objeto especifico por ejemplo cubo azul o cubo rojo o cubo verde o bowl unicamente.
        #     - SOLAMENTE SI la solicitud del usuario indica que quiere tomar un objeto en especifico (cubo azul, cubo rojo, cubo verde o bowl unicamente) debes ejecutar "tool_agent_take_objects" que se encargara de tomar el objeto en la simulacion.
        #     Al finalizar la ejecucion de la funcion "tool_agent_take_objects" retorna al usuario que el obejto especifico fue tomado por el robot, ahora pregunta si quiere que realice alguna accion, como soltarlo en un lugar especifico por ejemplo.
        #         - SOLAMENTE SI el usuario solicita soltar el objeto en un lugar especificado ejecuta la funcion "agent_drop_objects" para realizar esa accion. Al final de la ejecucion de esa funcion, pregunta al usuaio si desea continuar experimentando.
        # Si el Usuario consulta cual es el objetivo de este proyecto debes contestar que es un proyecto que permite interconectar la complejidad de la programacion de robots, en este caso un robot SCARA,
        # con la parametrizacion de un LLM para funcionamiento como un multiagente con funciones especificas cada uno. El alcance de esto es ayudar y facilitar las tareas de personas que tienen problemas de movilidad, programacion de tareas repetitivas,
        # apoyo logistico para organizacion y despacho de productos, entre otras muchas funciones.
        # """

        system_prompt = """
        You are an agent in charge of guiding users in an interface that shows 3 views of a SCARA robot scene and various objects from a CoopeliaSim simulation. Users can interact with the simulator through you. To do this, they must define if they want to pick up a specific object, for example, a blue cube, a red cube, a green cube, or a bowl only.
        - ONLY IF the user's request indicates that they want to pick up a specific object (blue cube, red cube, green cube, or bowl only) you must execute "tool_agent_take_objects" which will be in charge of picking up the object in the simulation.
        At the end of the execution of the "tool_agent_take_objects" function, it returns to the user that the specific object was picked up by the robot. Now it asks if they want to perform any action, such as dropping it in a specific place, for example.
        - ONLY IF the user requests to drop the object in a specified place, execute the "agent_drop_objects" function to perform that action. At the end of the execution of that function, ask the user if he/she wants to continue experimenting.
        If the user asks what the objective of this project is, you must answer that it is a project that allows the complexity of robot programming to be interconnected, in this case a SCARA robot,
        with the parameterization of an LLM to function as a multi-agent with specific functions each. The scope of this is to help and facilitate the tasks of people who have mobility problems, programming of repetitive tasks,
        logistical support for the organization and dispatch of products, among many other functions.
        """


        agent = FunctionCallingAgent.from_tools(
            query_engine_tools,
            llm=llm,
            verbose=True,
            allow_parallel_tool_calls=False,
            system_prompt=system_prompt
        )

        st.session_state.agent = agent

def image_to_base64(path_image):
    image = Image.open(path_image)
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return img_str

def get_postion_final_object(name_specific_object: str) -> list:
    """Get object postion from CoppeliaSim scene to move efector final"""
    _, Rectangle=sim.simxGetObjectHandle(st.session_state.clientID, name_specific_object, sim.simx_opmode_blocking)  # Obejct to move final effector
    _,postion_object=sim.simxGetObjectPosition(st.session_state.clientID, Rectangle, -1, sim.simx_opmode_blocking) # Position obejct to move final effector
    return [postion_object[0], postion_object[1], postion_object[2]+0.05]

def calculate_distance_r(px: float, py: float) -> float:
    """Calculate distance between 2 points at 2D dimention plane (first point is located in (0, 0 ))"""
    r = math.sqrt(math.pow(px, 2) + math.pow(py, 2))
    return r

def calculate_beta_angle(l2: float, l3: float, r: float) -> float:
    """Calculate angle beta to calculate main angles"""
    try:
        beta = math.acos((math.pow(l2, 2)+math.pow(r, 2)-math.pow(l3, 2))/(2*l2*r))
        return beta
    except ValueError:
        beta = 0
        return beta
    
def calculate_angle_q1(px: float, py: float, beta: float) -> float:
    """Calculate q1 angle in degres"""
    q1 = math.atan2(py, px) - beta
    q1_grados = math.degrees(q1)
    return q1

def calculate_angle_q2(r: float, l3: float, beta: float, q1: float) -> float:
    """Calculate q2 angle  in degres"""
    q2 = q1 + math.asin((r/l3)*math.sin(beta))
    q2_grados = math.degrees(q2)
    return q2

def calculate_distance_q3(l1: float, pz: float) -> float:
    """Calculate q3 distance in meters"""
    q3 = l1 - pz
    return q3

def set_postion_q1(q1: float) -> list:
    """set position to joint q1 from the robot using angle q1 calculated"""
    _, MTB_axis1=sim.simxGetObjectHandle(st.session_state.clientID, 'MTB_axis1', sim.simx_opmode_blocking)  # Joint revolution 1 (q1)
    _ = sim.simxSetObjectOrientation(st.session_state.clientID, MTB_axis1, -1 , (0, 0, q1), sim.simx_opmode_oneshot)  # Euler angles (alpha, beta, gamma)
    _, orientation_MTB_axis1=sim.simxGetObjectOrientation(st.session_state.clientID, MTB_axis1, -1, sim.simx_opmode_blocking)
    return orientation_MTB_axis1[2]

def set_postion_q2(q2: float) -> list:
    """set position to joint q2 from the robot using angle q2 calculated"""
    _, MTB_axis2=sim.simxGetObjectHandle(st.session_state.clientID, 'MTB_axis2', sim.simx_opmode_blocking)  # Joint revolucion 2 (q2)
    _ = sim.simxSetObjectOrientation(st.session_state.clientID, MTB_axis2, -1 , (0, 0, q2), sim.simx_opmode_oneshot)  # Euler angles (alpha, beta, gamma)
    _, orientation_MTB_axis2=sim.simxGetObjectOrientation(st.session_state.clientID, MTB_axis2, -1, sim.simx_opmode_blocking)
    return orientation_MTB_axis2[2]

def set_postion_q3(q3: float) -> list:
    """set position to joint q3 from the robot using distance q3 calculated"""
    _, MTB_axis3=sim.simxGetObjectHandle(st.session_state.clientID, 'MTB_axis3', sim.simx_opmode_blocking)  # Joint prismatic 3 (q3)
    _,position_MTB_axis3=sim.simxGetObjectPosition(st.session_state.clientID, MTB_axis3, -1, sim.simx_opmode_blocking)
    position_MTB_axis3[-1] = q3
    _ = sim.simxSetObjectPosition(st.session_state.clientID, MTB_axis3, -1, tuple(position_MTB_axis3), sim.simx_opmode_oneshot)  # Distances en (x, y, z)
    _, position_MTB_axis3=sim.simxGetObjectPosition(st.session_state.clientID, MTB_axis3, -1, sim.simx_opmode_blocking)
    return position_MTB_axis3[2]

def get_distance_prooximitySensor() -> int:
    time.sleep(0.5)
    _, suctionPadSensor=sim.simxGetObjectHandle(st.session_state.clientID, 'Proximity_sensor', sim.simx_opmode_blocking)  # Proximity sensor
    _, _, point, obj, n = sim.simxReadProximitySensor(st.session_state.clientID, suctionPadSensor, sim.simx_opmode_blocking)

    if point[2] < 0.01:
        return obj
    else:
        return None

def take_object(obj: int) -> None:
    _, suctionPad=sim.simxGetObjectHandle(st.session_state.clientID, 'suctionPad', sim.simx_opmode_blocking)  # suctionPad
    sim.simxSetObjectParent(st.session_state.clientID, obj, suctionPad, True, sim.simx_opmode_blocking)
    st.session_state.number_objectParent = obj

def drop_object() -> None:
    _ = sim.simxSetObjectParent(st.session_state.clientID, st.session_state.number_objectParent, -1, True, sim.simx_opmode_blocking)
    st.session_state.number_objectParent = None

def on_click_callback():
    if st.session_state.clientID is None:
        st.session_state.clientID = connect_coppeliaSim()
        if st.session_state.clientID is not None:
            with col2_c1:
                st.info("Connected with CoppeliaSim", icon="ℹ️")

                with st.spinner("Parsing text input..."):
                    human_prompt = st.session_state.human_prompt
                    st.session_state.history.append(Message("human", human_prompt))
                    response = st.session_state.agent.chat(human_prompt)
                    st.session_state.history.append(Message("ai", response))
                    st.session_state.human_prompt = ""
                    get_image('CAM_1')
                    get_image('CAM_2')
                    get_image('CAM_3')

        else:
            with col2_c1:
                st.warning("You must initialize CoppeliaSim for it to work", icon="ℹ️")

def load_css():
    with open(".streamlit/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

initialize_session()

st.set_page_config(layout='wide', initial_sidebar_state="expanded", page_title="LLRAM", page_icon="💡")

# load_css()

images = ["static/CAM_1.png", "static/CAM_2.png", "static/CAM_3.png"]

hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

col1_c1, col2_c1 = st.columns([2, 1])

with col1_c1:

    c1_1 = st.container()

    with c1_1:
        col1_c1_1, col2_c1_1 = st.columns([1, 3])

        with col1_c1_1:
            st.markdown(
                f"""

                <div style="text-align: center; color: #73c7f1; font-size: 30px; font-weight: bold;">
                    LLRAM
                </div>
                <div style="text-align: center; color: #73c7f1; font-size: 18px; font-weight: bold;">
                    (Large Language Robotic Arm Model)
                </div>

                """, unsafe_allow_html=True)
        
        with col2_c1_1:
            st.markdown(
                """
                <div style="text-align: justify; padding: 5px; margin: 0px 20px 0px">
                    <p>The Large Language Robotic Arm Model (LLRM) is an advanced artificial intelligence system that allows users to interact with the robot using natural language commands. 
                    Featuring natural language processing (NLP), computer vision, and both direct and inverse kinematics, the LLRM translates verbal or textual instructions into precise 
                    actions while visually interpreting its environment. Ideal for applications in manufacturing and automation, it enhances efficiency and adapts to various tasks based 
                    on user needs.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    c1_2 = st.container()

    with c1_2:
        col1_c1_2, col2_c1_2 = st.columns([1, 1])

        with col1_c1_2:   
            st.markdown(
                f"""
                <div class="image-container">
                    <img src="data:image/png;base64,{image_to_base64(images[0])}" width="600" height="450">
                </div>
                """, unsafe_allow_html=True)
            
        with col2_c1_2:
            # Mostrar la imagen actual
            st.markdown(
                f"""
                <div class="image-container">
                    <img src="data:image/png;base64,{image_to_base64(images[1])}" width="600" height="450">
                </div>
                """, unsafe_allow_html=True)
            
        st.write("")
            
        st.markdown(
            f"""
            <div class="image-container">
                <img src="data:image/png;base64,{image_to_base64(images[2])}" width="600" height="450">
            </div>
            """, unsafe_allow_html=True)   

with col2_c1:

    paper_chat_style = """
            {
                margin: 20px;
                background-color: #bac2c9;
                padding: 1.5em;  
                border-radius: 1.5em;
                border: 0.5px solid #bac2c9;
                box-sizing: border-box;
            }
            """
    markdown_style = """
            .stMarkdown {
                padding-right: 1.5em;
                padding-left: 1.5em;
            }
            """

    with stylable_container(key="chat_container", css_styles=[paper_chat_style, markdown_style]):
        st.markdown("<p style='font-size:20px; text-align:center; color: #666e74;\
                    font-weight:bold;font-style:italic'>LLRAM Agent 🤖</p>", unsafe_allow_html=True)

        chat_placeholder = st.container()
        with chat_placeholder:               
            for chat_ph in st.session_state.history:
                
                div = f"""
                    <div class="chat-row 
                        {'' if chat_ph.origin == 'ai' else 'row-reverse'}">
                        <img class="chat-icon" src="data:image/png;base64,{image_to_base64("static/ai_icon.png")  
                            if chat_ph.origin == 'ai' else image_to_base64("static/user_icon.png")}" width=32 height=32>
                        <div class="chat-bubble
                        {'ai-bubble' if chat_ph.origin == 'ai' else 'human-bubble'}">
                            &#8203;{chat_ph.message}
                        </div>
                    </div>
                """
                st.markdown(div, unsafe_allow_html=True)

        prompt_placeholder = st.form("chat-form")
        with prompt_placeholder:
            cols = st.columns((6, 1))
            cols[0].text_input(
                "Chat",
                value="",
                placeholder="Escribe tu consulta",
                label_visibility="collapsed",
                key="human_prompt",
            )
            cols[1].form_submit_button(
                "➤", 
                type="primary",
                on_click=on_click_callback, 
            )

