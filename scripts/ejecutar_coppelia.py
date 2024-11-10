# import sim
# import time
# import subprocess

# # Paso 1: Iniciar CoppeliaSim desde Python (opcional)
# # Abre CoppeliaSim si no está ejecutándose
# subprocess.Popen(r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe", shell=True)

# # Paso 2: Conectar a la simulación de CoppeliaSim
# def connect_to_coppelia():
#     port = 19999  # Puerto predeterminado para la conexión remota
#     sim.simxFinish(-1)  # Cerrar conexiones anteriores
#     clientID = sim.simxStart('127.0.0.1', port, True, True, 2000, 5)  # Iniciar conexión
#     if clientID != -1:
#         print("Conectado a CoppeliaSim")
#         return clientID
#     else:
#         print("No se pudo conectar a CoppeliaSim")
#         return None

# # Paso 3: Cargar una escena y ejecutarla
# def load_and_run_scene(clientID, scene_path):
#     # Cargar la escena
#     sim.simxLoadScene(clientID, scene_path, 0, sim.simx_opmode_blocking)
#     print(f"Escena {scene_path} cargada")
    
#     # Iniciar la simulación
#     sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
#     print("Simulación iniciada")

# # Usar las funciones
# clientID = connect_to_coppelia()

# print("A")

# if clientID != None:
#     # Ruta a tu escena (ajusta la ruta según tu sistema operativo)
#     scene_path = r"C:\Users\matea\OneDrive\Documentos\Mateo\Proyectos\concurso NVIDIA\brazo_robot.ttt"
    
#     # Cargar y ejecutar la escena
#     load_and_run_scene(clientID, scene_path)

#     # Mantener la simulación en ejecución durante algunos segundos
#     time.sleep(30)

#     # # Detener la simulación
#     # sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
#     # print("Simulación detenida")
    
#     # # Cerrar la conexión
#     # sim.simxFinish(clientID)
#     # print("Conexión cerrada")

# import subprocess

# # Ruta al ejecutable de CoppeliaSim y al archivo de escena
# coppelia_path = r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe"
# scene_path = r"C:\Users\matea\OneDrive\Documentos\Mateo\Proyectos\concurso NVIDIA\brazo_robot.ttt"

# # Comando que quieres ejecutar
# command = [coppelia_path, "-s5000", "-q", scene_path]

# # Ejecutar el comando
# subprocess.Popen(command, shell=True)

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')
print(sim.getStringParam(sim.stringparam_scenedefaultdir) + 'brazo_robot.ttt')
# sim.loadScene(sim.getStringParam(sim.stringparam_scenedefaultdir) + 'brazo_robot.ttt')

# sim.setStepping(True)

# sim.startSimulation()

# port=19999
# sim.simxFinish(-1) 
# clientID=sim.simxStart('127.0.0.1',port,True,True,2000,5)
# # sim.simxStopSimulation(clientID, )
# if clientID == 0: 
#     print("conectado a", port)
#     code, MTB_Robot=sim.simxGetObjectHandle(clientID, 'MTB_Robot', sim.simx_opmode_blocking)  # Robot completo
#     print(code, MTB_Robot)
#     code = sim.simxSetObjectPosition(clientID, MTB_Robot, -1, (0,0,0.5), sim.simx_opmode_oneshot)  # Distancias en (x, y, z)
#     print(code)
#     code, orientation_MTB_Robot=sim.simxGetObjectOrientation(clientID, MTB_Robot, -1, sim.simx_opmode_blocking)
#     print(code, orientation_MTB_Robot)
# else:
#     print("No hay conexion")

while (t := sim.getSimulationTime()) < 5:
    MTB_Robot = sim.getObject('/MTB')
    # sim.
    sim.step()
sim.stopSimulation()