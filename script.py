import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.constants as const
import time

# --- 1. Definição dos Campos Eletromagnéticos ---

def get_E_field(position):
    """
    Retorna o vetor do campo elétrico E na_posição_dada.
    Mantido como zero para focar no campo magnético.
    """
    return np.array([0.0, 0.0, 0.0])

def get_B_field(position):
    """
    Retorna o vetor do campo magnético B na_posição_dada.
    Implementa um campo quadrupolo magnético ideal (foco em Y, desfoco em X).
    Assume que a partícula se move primariamente ao longo de Z.
    B_x = g * y
    B_y = g * x
    B_z = 0
    """
    g = 50.0  # Gradiente do campo quadrupolo (Tesla / metro) - Ajuste conforme necessário
    x, y, z = position
    
    B_x = g * y
    B_y = g * x
    B_z = 0.0
    
    return np.array([B_x, B_y, B_z])

# --- 2. Definição da Partícula ---

# Múon (μ-)
particle_mass = particle_mass = 1.883531627e-28  # Massa em kg (aprox. 1.88e-28 kg)
particle_charge = -const.elementary_charge  # Carga em Coulombs (negativa)

# --- 3. Condições Iniciais e Parâmetros da Simulação ---

# Posição inicial [x, y, z] em metros
# Começa um pouco fora do eixo X para sentir o campo quadrupolo
initial_position = np.array([0.01, 0.0, 0.0]) 

# Velocidade inicial [vx, vy, vz] em m/s
# 50% da velocidade da luz na direção Z (direção do feixe)
initial_velocity = np.array([0.0, 0.0, 0.5 * const.c]) 

# Parâmetros de tempo e Precisão (Aumentados)
dt = 1.0e-12  # Passo de tempo (bem pequeno para alta precisão com B_quad e drag)
total_time = 5.0e-6  # Tempo total da simulação (aumentado para trajetória longa)
n_steps = int(total_time / dt) # Isso dará 5 milhões de passos

# --- 4. Parâmetro de Arrasto ---
# Coeficiente de arrasto (k). Este valor é *hipotético* e precisa ser 
# ajustado para ver um efeito razoável. Um valor pequeno é necessário
# devido às altas velocidades e pequena massa.
drag_coefficient = 1e-21  # kg/s (valor inicial, ajuste se necessário)

# --- 5. Função de Aceleração (Lorentz + Arrasto) ---

def calculate_acceleration(charge, mass, position, velocity):
    """Calcula a aceleração usando Lorentz + Força de Arrasto."""
    E = get_E_field(position)
    B = get_B_field(position)
    
    # Força de Lorentz
    force_lorentz = charge * (E + np.cross(velocity, B))
    
    # Força de Arrasto (F_drag = -k * v)
    force_drag = -drag_coefficient * velocity
    
    # Força Total
    total_force = force_lorentz + force_drag
    
    # Aceleração (a = F / m)
    acceleration = total_force / mass
    return acceleration

# --- 6. O Simulador (Loop de Integração RK4) ---

def simulate_particle_trajectory():
    """Executa a simulação e retorna a trajetória."""
    
    print(f"Iniciando simulação com {n_steps} passos...")
    print("Isso pode levar alguns minutos, por favor aguarde...")

    positions = np.zeros((n_steps + 1, 3), dtype=np.float64)
    velocities = np.zeros((n_steps + 1, 3), dtype=np.float64)

    positions[0] = initial_position
    velocities[0] = initial_velocity

    start_time = time.time()

    for i in range(n_steps):
        r = positions[i]
        v = velocities[i]

        # RK4
        a1 = calculate_acceleration(particle_charge, particle_mass, r, v)
        k1_r = v; k1_v = a1

        a2 = calculate_acceleration(particle_charge, particle_mass, r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v)
        k2_r = v + 0.5 * dt * k1_v; k2_v = a2

        a3 = calculate_acceleration(particle_charge, particle_mass, r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v)
        k3_r = v + 0.5 * dt * k2_v; k3_v = a3

        a4 = calculate_acceleration(particle_charge, particle_mass, r + dt * k3_r, v + dt * k3_v)
        k4_r = v + dt * k3_v; k4_v = a4

        positions[i+1] = r + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        velocities[i+1] = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        # Barra de progresso (menos frequente para não atrasar)
        if (i + 1) % (n_steps // 100) == 0:
            print(f"Progresso: {((i+1)/n_steps)*100:.0f}%")

    end_time = time.time()
    print(f"Simulação concluída em {end_time - start_time:.2f} segundos.")
    
    return positions, velocities # Retorna também as velocidades para análise posterior

# --- 7. Execução e Visualização ---

trajectory, final_velocities = simulate_particle_trajectory()

x = trajectory[:, 0]
y = trajectory[:, 1]
z = trajectory[:, 2]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plota a trajetória com linha mais fina para muitos pontos
ax.plot(x, y, z, label='Trajetória do Múon', lw=0.8) # lw = line width

ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Início', depthshade=True)
ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='Fim', depthshade=True)

ax.set_title('Simulação de Múon em Campo Quadrupolo com Arrasto')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
ax.grid(True)

# Ajuste automático dos limites para ver a trajetória inteira
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range*0.55, mid_x + max_range*0.55)
ax.set_ylim(mid_y - max_range*0.55, mid_y + max_range*0.55)
ax.set_zlim(mid_z - max_range*0.55, mid_z + max_range*0.55)

# Força a escala a ser igual nos eixos X e Y para ver o foco/desfoco
ax.set_aspect('auto') # 'auto' é padrão, 'equal' pode distorcer Z demais

plt.show()

# Opcional: Plotar a velocidade para ver o efeito do arrasto
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# speeds = np.linalg.norm(final_velocities, axis=1)
# time_points = np.linspace(0, total_time, n_steps + 1)
# ax2.plot(time_points, speeds / const.c)
# ax2.set_title('Velocidade da Partícula (em % de c) vs. Tempo')
# ax2.set_xlabel('Tempo (s)')
# ax2.set_ylabel('Velocidade / c')
# ax2.grid(True)
# plt.show()