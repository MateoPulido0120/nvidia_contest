import streamlit as st
import time
from PIL import Image
import numpy as np
import sim
import warnings
from io import BytesIO
import base64

# Desactivar warnings de deprecación
warnings.filterwarnings("ignore", category=DeprecationWarning)

def connect_coppeliaSim():
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

def get_objectsCoppelia(clientID):
    _, MTB_Robot=sim.simxGetObjectHandle(clientID, 'MTB_Robot', sim.simx_opmode_blocking)  # Robot completo
    _, MTB_axis1=sim.simxGetObjectHandle(clientID, 'MTB_axis1', sim.simx_opmode_blocking)  # Junta revolucion 1
    _, MTB_axis2=sim.simxGetObjectHandle(clientID, 'MTB_axis2', sim.simx_opmode_blocking)  # Junta revolucion 2
    _, MTB_axis3=sim.simxGetObjectHandle(clientID, 'MTB_axis3', sim.simx_opmode_blocking)  # Junta prismatica 3
    _, MTB_axis4=sim.simxGetObjectHandle(clientID, 'MTB_axis3', sim.simx_opmode_blocking)  # Junta revolucion 4

    _, suctionPadSensor=sim.simxGetObjectHandle(clientID, 'Proximity_sensor', sim.simx_opmode_blocking)  # Sensor de proximidad
    _, suctionPad=sim.simxGetObjectHandle(clientID, 'suctionPad', sim.simx_opmode_blocking)  # Junta revolucion 4

    return MTB_Robot, MTB_axis1, MTB_axis2, MTB_axis3, MTB_axis4, suctionPadSensor, suctionPad


def move_griperCoppelia(type_movement, clientID, MTB_Robot, MTB_axis3, suctionPadSensor, suctionPad):
    _,orientation_MTB_axis3=sim.simxGetObjectPosition(clientID, MTB_axis3, -1, sim.simx_opmode_blocking)

    if type_movement == "down":
        orientation_MTB_axis3[-1]+=-0.02
    else:
        orientation_MTB_axis3[-1]+=0.02

    _ = sim.simxSetObjectPosition(clientID, MTB_axis3, -1, tuple(orientation_MTB_axis3), sim.simx_opmode_oneshot)  # Distancias en (x, y, z)
    _, orientation_MTB_Robot=sim.simxGetObjectOrientation(clientID, MTB_Robot, -1, sim.simx_opmode_blocking)
    
    
    res, dist, point, obj, n = sim.simxReadProximitySensor(clientID, suctionPadSensor, sim.simx_opmode_blocking)

    print(res, dist, point, obj, n)

    if point[2] < 0.01:
        print("Colision")
        sim.simxSetObjectParent(clientID, obj, suctionPad, True, sim.simx_opmode_blocking)
        st.session_state.number_objectParent = obj

def get_simulation_image():
    _, vision_sensor1 =sim.simxGetObjectHandle(st.session_state.clientID, 'Vision_sensor', sim.simx_opmode_blocking)  # Sensor de vision 1
    returnCode, resolution, image = sim.simxGetVisionSensorImage(st.session_state.clientID, vision_sensor1, 0, sim.simx_opmode_blocking)

    # Convertimos la lista a un arreglo de numpy
    imagen_array = np.array(image, dtype=np.uint8)

    # Asegúrate de que la forma del arreglo es (256, 256, 3)
    imagen_array = imagen_array.reshape((256, 256, 3))

    imagen_array = np.flipud(np.fliplr(imagen_array))

    # Guardamos la imagen en formato RGB usando PIL
    imagen = Image.fromarray(imagen_array, mode='RGB')

    return imagen

def image_to_base64(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return img_str

def initialize_session():
    if 'clientID' not in st.session_state:
        st.session_state.clientID = connect_coppeliaSim()
    if 'number_objectParent' not in st.session_state:
        st.session_state.number_objectParent = None
    if "type_movement" not in st.session_state:
        st.session_state.type_movement = None
    if "gripper_load" not in st.session_state:
        st.session_state.gripper_load = False



# Configuración de Streamlit
st.title("Simulación en Tiempo Real con Control del Robot")
st.text("Usa los botones para mover el robot y observa la simulación en tiempo real.")


initialize_session()


# Crear botones para mover el robot
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("down"):
        st.session_state.type_movement = "down"
with col2:
    if st.button("up"):
        st.session_state.type_movement = "up"
with col3:
    if st.button("soltar"):
        st.session_state.gripper_load = True

# Contenedor para el streaming
placeholder = st.empty()

while True:
        
    # Control de movimiento según el botón presionado
    if st.session_state.type_movement:
        if st.session_state.clientID is not None:
            MTB_Robot, MTB_axis1, MTB_axis2, MTB_axis3, MTB_axis4, suctionPadSensor, suctionPad = get_objectsCoppelia(st.session_state.clientID)
            move_griperCoppelia(st.session_state.type_movement, st.session_state.clientID, MTB_Robot, MTB_axis3, suctionPadSensor, suctionPad)
            st.session_state.type_movement = None

    if st.session_state.gripper_load:
        if st.session_state.clientID is not None:
            MTB_Robot, MTB_axis1, MTB_axis2, MTB_axis3, MTB_axis4, suctionPadSensor, suctionPad = get_objectsCoppelia(st.session_state.clientID)
            returnCode = sim.simxSetObjectParent(st.session_state.clientID, st.session_state.number_objectParent, -1, True, sim.simx_opmode_blocking)
            st.session_state.gripper_load = False

    image = get_simulation_image()
    
    # Mostrar la imagen en el contenedor de Streamlit
    placeholder.image(image, use_column_width=True)
