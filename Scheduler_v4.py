import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Request parameters
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": -6.37,
    "longitude": 106.83,
    "hourly": ["temperature_2m", "shortwave_radiation", "rain", "cloud_cover"],
    "timezone": "auto",
    "forecast_days": 1
}
responses = openmeteo.weather_api(url, params=params)

# Process first location
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Hourly variables
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_shortwave_radiation = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()

# Create datetime index in UTC
date_range = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)

# Create dataframe
hourly_data = {
    "date": date_range,
    "temperature_2m": hourly_temperature_2m,
    "shortwave_radiation": hourly_shortwave_radiation,
    "rain": hourly_rain,
    "cloud_cover": hourly_cloud_cover
}
hourly_dataframe = pd.DataFrame(data=hourly_data)

# Convert timezone to local (e.g. Jakarta)
hourly_dataframe["date"] = hourly_dataframe["date"].dt.tz_convert("Asia/Jakarta")

####################
# SCHEDULER
####################

import pandas as pd
import gspread
import os
from google.colab import drive
drive.mount('/content/drive')
# from dotenv import load_dotenv
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from collections import defaultdict
from pyomo.environ import (
    ConcreteModel, Set, Var, Binary, NonNegativeReals, Param,
    Constraint, Objective, maximize, SolverFactory, value, NonNegativeIntegers
)
from pyomo.util.infeasible import log_infeasible_constraints

import pandas as pd
import gspread
import os
from google.colab import drive
drive.mount('/content/drive')
# from dotenv import load_dotenv
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from collections import defaultdict
from pyomo.environ import (
    ConcreteModel, Set, Var, Binary, NonNegativeReals, Param,
    Constraint, Objective, maximize, SolverFactory, value, NonNegativeIntegers
)
from pyomo.util.infeasible import log_infeasible_constraints

# ---------------
# Parameters
# ---------------

# Numbers of Battery Slots
N_SLOTS = 8

# Battery Capacity in kWh
BATTERY_KWH = 2.7

# Maximum Power of Charging in kW
P_CHARGE_MAX = 0.8

# Maximum Power of disharging in kW
P_DISCHARGE_MAX = 0.8

# Grid electricity purchase price (IDR per kWh)
PLN_PRICE = 1035

# Revenue per battery swap (IDR)
SWAP_PRICE = 50000

# Revenue for solar surplus sold (multiplied by PLN_PRICE)
SOLAR_SELL = 1.2 * PLN_PRICE

# Degradation cost per kWh throughput (IDR)
DEGR_COST = 150

# Store consumption (kW) during open hours
LOAD_OPEN_KW = 0

# Store consumption (kW) during closed hours
LOAD_CLOSED_KW = 2.3

# Store open hours
OPEN_HOURS = range(8, 22)

# Battery Condition State after swap, dummy is 0
BATTERY_CONDITION_STATE = 0.0

# PV Panel Parameters
PV_PANEL_WP = 300    # watts-peak per panel
PV_PANEL_COUNT = 9   # number of panels

# Irradiance per hour (W/m2)
IRRADIANCE = dict(hourly_dataframe["shortwave_radiation"])

# Recorded swap timestamps
# Average by Historical Swap Times, let say 1 week historical swap times
SWAP_TIMES = """
07:25:32 08:33:17 09:11:08 09:11:41 09:20:38 09:23:15 09:24:15
09:29:56 09:45:23 09:49:07 09:50:43 12:03:36 12:05:49 12:09:11
12:29:04 12:38:35 12:44:24 15:41:29 18:04:06 18:06:30 18:10:55
18:13:27 18:14:44 18:15:36 18:20:19 18:37:48 18:45:00 19:46:50
21:42:48 21:49:30 22:05:01 22:14:03
""".split()

# Times
HOURS = range(24)
swap_hour = defaultdict(int)
for t in SWAP_TIMES:
    swap_hour[int(t.split(":")[0])] += 1

# PV Power
pv_kw = {
    h: (IRRADIANCE.get(h,0)/1000)*(PV_PANEL_WP*0.001)*PV_PANEL_COUNT
    for h in HOURS
}

# Power needed to discharging to store
store_kW = {h: LOAD_OPEN_KW if h in OPEN_HOURS else LOAD_CLOSED_KW for h in HOURS}

# Threshold battery capacity for battery could be swapped
threshold = BATTERY_KWH * 0.5

# Big M Notation
M = BATTERY_KWH + P_CHARGE_MAX + P_DISCHARGE_MAX

# ---------------
# DECISOIN VARIABLE
# ---------------

# Initializing the Model
model = ConcreteModel()

# # Define battery slot indices (0 to N_SLOTS-1)
model.slots = Set(initialize= range(N_SLOTS))

# Define times indices
model.times = Set(initialize= HOURS)

# Charge Variable (Binary)
model.charge = Var(model.slots, model.times, domain= Binary)

# Discharge Variable (Binary)
model.discharge = Var(model.slots, model.times, domain = Binary)

# Swap condition in given time (Binary)
model.swap = Var(model.slots, model.times, domain = Binary)

# State of Charge - SOC variable for each battery in each slot
model.soc = Var(model.slots, model.times, domain = NonNegativeReals, bounds= (0, BATTERY_KWH))

# State of Battery whether each of those needs charge or not
model.needs_charge = Var(model.slots, model.times, domain = Binary)

# If Swap success in given slot and time, it called swaphit
model.swaphit = Var(model.times, domain = NonNegativeIntegers, bounds= (0, N_SLOTS))

# If Swap failed in given slot and time, it called unserved swap
model.unserved_swap = Var(model.times, domain = NonNegativeIntegers, bounds= (0, N_SLOTS))

# Power of Grid which could be drawn for either discharing or charging
model.grid_kWh = Var(model.times, domain = NonNegativeReals)

# Excessive Solar Power (Not swapped nor discharged)
model.excess_solar = Var(model.times, domain = NonNegativeReals)

# Model Idle
model.idle = Var(model.slots, model.times, domain = Binary)

# Threshold parameter 
model.threshold = Param(initialize=threshold)

# Battery Readiness
model.battery_ready = Var(model.slots, model.times, within=Binary)

# ---------------
# Objective
# ---------------

# Revenue for all swaps that happened
revenue_swap = 50000 * sum(model.swaphit[t] for t in HOURS)

# Revenue from Excessed Solar
revenue_solar = SOLAR_SELL * sum(max(0, pv_kw[t] - store_kW[t]) for t in HOURS)

# Cost to buy electricity from grid to discharged to store
cost_grid = PLN_PRICE * sum(model.grid_kWh[t] for t in HOURS)

# Cost from battery degradation due to full swaps and discharges
# cost_degr = DEGR_COST * (
#     BATTERY_KWH * sum(model.swap[i, t] for i in model.slots for t in model.times) +
#     P_DISCHARGE_MAX * sum(model.discharge[i, t] for i in model.slots for t in model.times)
# )

# Penalty for unserved swaps
penalty_unserved = 100000 * sum(model.unserved_swap[t] for t in HOURS)

# Model Objective
model.maximize_profit = Objective (expr = revenue_swap + revenue_solar - cost_grid - penalty_unserved,
sense = maximize
)

# --------------------------
# Constraint
# --------------------------

# Initial Simulation (Akan dihapus saat real world scenario sudah berjalan)
def initial_soc_fix_rule(model, s):
    # Memaksa SoC untuk setiap slot 's' pada waktu pertama ('model.times.first()')
    # agar sama dengan kapasitas penuh baterai (BATTERY_KWH).
    return model.soc[s, model.times.first()] == BATTERY_KWH

model.initial_soc_constraint = Constraint(model.slots, rule=initial_soc_fix_rule)

# 1. Exclusivity, where only one of charge/discharge/swap could happen at the time (charge + discharge + swap <= 1)
def exclusive_actions_rule(model, slots, times):
    return model.charge[slots, times] + model.discharge[slots, times] + model.idle[slots, times] <= 1
model.exclusive_actions = Constraint(model.slots, model.times, rule= exclusive_actions_rule)

# 2. SOC = soc of previous time + charging power * charge condition - discharging power * discharge condition - battery max capacity *  model swap in a given slot and times
# Constraint 2.1.: Enforce dynamic update if no swap (model.swap == 0)
# This constraint ensures: model.soc[s,t] >= (dynamic_RHS) when swap=0
def soc_dyn_if_no_swap_lower_rule(model, slots, times):
    if times == model.times.first():
        return Constraint.Skip
    time_previous = model.times.prev(times)
    dynamic_RHS = (
        model.soc[slots, time_previous] +
        P_CHARGE_MAX * model.charge[slots, times] -
        P_DISCHARGE_MAX * model.discharge[slots, times]
    )
    # When model.swap[s,t] = 0 (no swap):
    #   RHS becomes dynamic_RHS - M_BIG_SOC_DYN * 0 = dynamic_RHS. Constraint: model.soc[s,t] >= dynamic_RHS. (Binding)
    # When model.swap[s,t] = 1 (swap occurs):
    #   RHS becomes dynamic_RHS - M_BIG_SOC_DYN * 1 = dynamic_RHS - M_BIG_SOC_DYN.
    #   Since model.soc[s,t] is always >= 0, and dynamic_RHS - M_BIG_SOC_DYN is a very large negative number,
    #   this constraint (model.soc[s,t] >= very_large_negative_number) becomes non-binding.
    return model.soc[slots, times] >= dynamic_RHS - M * model.swap[slots, times]
model.soc_dyn_if_no_swap_lower = Constraint(model.slots, model.times, rule=soc_dyn_if_no_swap_lower_rule)

# Constraint 2.2.: Enforce dynamic update if no swap (model.swap == 0)
# This constraint ensures: model.soc[s,t] <= (dynamic_RHS) when swap=0
def soc_dyn_if_no_swap_upper_rule(model, slots, times):
    if times == model.times.first():
        return Constraint.Skip
    time_previous = model.times.prev(times)
    dynamic_RHS = (
        model.soc[slots, time_previous] +
        P_CHARGE_MAX * model.charge[slots, times] -
        P_DISCHARGE_MAX * model.discharge[slots, times]
    )
    return model.soc[slots, times] <= dynamic_RHS + M * model.swap[slots, times]
model.soc_dyn_if_no_swap_upper = Constraint(model.slots, model.times, rule=soc_dyn_if_no_swap_upper_rule)

# Constraint 2.3.: Enforce SoC reset if swap occurs (model.swap == 1)
# This constraint ensures: model.soc[s,t] >= BATTERY_CONDITION_STATE when swap=1
def soc_reset_if_swap_lower_rule(model, slots, times):
    if times == model.times.first():
        return Constraint.Skip
    return model.soc[slots, times] >= BATTERY_CONDITION_STATE - M * (1 - model.swap[slots, times])
model.soc_reset_if_swap_lower = Constraint(model.slots, model.times, rule=soc_reset_if_swap_lower_rule)


# Constraint 2.4.: Enforce SoC reset if swap occurs (model.swap == 1)
# This constraint ensures: model.soc[s,t] <= BATTERY_CONDITION_STATE when swap=1
def soc_reset_if_swap_upper_rule(model, slots, times):
    if times == model.times.first():
        return Constraint.Skip
    return model.soc[slots, times] <= BATTERY_CONDITION_STATE + M * (1 - model.swap[slots, times])
model.soc_reset_if_swap_upper = Constraint(model.slots, model.times, rule=soc_reset_if_swap_upper_rule)


# 3. Piece Wise of needs charge if in prev times there were swapped happened or if SOC of previous times less than BATTERY_KWH
def need_charge_activation_rule(model, slots, times):
    if times == model.times.first():
        return Constraint.Skip

    time_previous = model.times.prev(times) # Previous TIme
    # M = BATTERY_KWH

    return model.needs_charge[slots, times] >= model.swap[slots, time_previous]
model.needs_charge_from_swap = Constraint(model.slots, model.times, rule=need_charge_activation_rule)

def need_charge_soc_rule(model, slots, times):
    if times == model.times.first():
        return Constraint.Skip

    epsilon = 0.01
    return model.soc[slots, times] <= BATTERY_KWH - epsilon + M * (1 - model.needs_charge[slots, times])
model.needs_charge_from_soc = Constraint(model.slots, model.times, rule=need_charge_soc_rule)

# def need_charge_soc_rule(model, slots, times):
#     if times == model.times.first():
#         return Constraint.Skip
#     time_previous = model.times.prev(times) # Previous TIme
#     M = BATTERY_KWH
#     epsilon = 0.01

#     return model.needs_charge[slots, times] >= 1 - (model.soc[slots, time_previous] - BATTERY_KWH + epsilon) / M
# model.needs_charge_from_soc = Constraint(model.slots, model.times, rule=need_charge_soc_rule)

# 3.1. Force model.charge = 1 if model.needs_charge_from_soc = 1
def charge_activation_rule(model, slots, times):
    return model.charge[slots, times] >= model.needs_charge[slots, times]
model.charge_activation = Constraint(model.slots, model.times, rule=charge_activation_rule)

# 4. Atleast 1 slots emtpy for swapping
model.is_full = Var(model.slots, model.times, domain=Binary)

def is_full_logic_rule(model, slots, times):
    epsilon = 0.01  # margin under full capacity
    return model.soc[slots, times] >= BATTERY_KWH - epsilon - M * (1 - model.is_full[slots, times])
model.full_logic = Constraint(model.slots, model.times, rule=is_full_logic_rule) # This forces model.is_full[i, t] = 1 only when soc[i, t] ≥ BATTERY_KWH - epsilon

# def is_full_low(m, s, t):
#     M = BATTERY_KWH
#     epsilon = 0.01  # margin under full capacity
#     return m.soc[s, t] <= BATTERY_KWH - epsilon + M * m.is_full[s, t]
# model.is_full_low = Constraint(model.slots, model.times, rule=is_full_low)

# def minimum_empty_packs_rule(model, times):
#     return sum(model.is_full[i, times] for i in model.slots) <= N_SLOTS - 1
# model.min_empty_packs = Constraint(model.times, rule=minimum_empty_packs_rule)

# 5. Total swap hit and swap unserved in a given time shouldnt more than all swap happened in the given time
def swap_fulfillment_rule(model, times):
    return model.swaphit[times] + model.unserved_swap[times] == swap_hour.get(times, 0)
model.swap_fulfillment = Constraint(model.times, rule= swap_fulfillment_rule)

# 6. Energy Balance: Solar + Discharge + Grid >= Store Load + Battery Charging
def energy_balance_rule(model, times):

    discharge_sum = sum(P_DISCHARGE_MAX * model.discharge[i, times] for i in model.slots)
    charge_sum = sum(P_CHARGE_MAX * model.charge[i, times] for i in model.slots)

    return pv_kw[times] + discharge_sum + model.grid_kWh[times] >= store_kW[times] + charge_sum
model.energy_balance = Constraint(model.times, rule = energy_balance_rule)

# 7. Computes unused solar beyond store load
def solar_init(model, t):
    return pv_kw.get(t, 0)  # Ensure `t` format matches keys in pv_kw

model.solar = Param(model.times, initialize=solar_init, within=NonNegativeReals)

def excess_solar_rule(model, times):
    return model.excess_solar[times] >= model.solar[times] - store_kW[times]
model.excess_solar_constraint = Constraint (model.times, rule= excess_solar_rule)

# 8. Constraint to link individual slot swaps to total successful swaps
def total_swaps_hit_rule(model, times):
    # The total number of successful swaps in 'swaphit' must equal
    # the sum of individual battery slots that are marked for a swap.
    return model.swaphit[times] == sum(model.swap[i, times] for i in model.slots)
model.total_swaps_hit = Constraint(model.times, rule=total_swaps_hit_rule)

# 9.
model.is_not_full = Var(model.slots, model.times, within=Binary)
eps = 1e-2
    
# Mbig dan eps sudah didefinisikan

# Kendala 1: Jika is_not_full == 1, maka SOC harus kurang dari BATTERY_KWH
# Ekspresi ini memastikan bahwa jika is_not_full = 1, maka SOC <= BATTERY_KWH - eps
def is_not_full_up_fixed_rule(m, s, t):
    return m.soc[s, t] <= BATTERY_KWH - eps + M * (1 - m.is_not_full[s, t])
model.is_not_full_up_fixed = Constraint(model.slots, model.times, rule=is_not_full_up_fixed_rule)

# Kendala 2: Jika is_not_full == 0, maka SOC harus sama dengan BATTERY_KWH
# Ekspresi ini memastikan bahwa jika is_not_full = 0, maka SOC >= BATTERY_KWH - eps
def is_not_full_low_fixed_rule(m, s, t):
    return m.soc[s, t] >= BATTERY_KWH - eps - M * m.is_not_full[s, t]
model.is_not_full_low_fixed = Constraint(model.slots, model.times, rule=is_not_full_low_fixed_rule)

#10.
# Constraint: No discharge during store open hours
def no_discharge_when_open_rule(model, s, t):
    if t in OPEN_HOURS:
        return model.discharge[s, t] == 0
    return Constraint.Skip

model.no_discharge_when_open = Constraint(model.slots, model.times, rule=no_discharge_when_open_rule)

#11. Idle Discharged Constraint
def idle_upper_charge_rule(m, s, t):
    # if (s, t) not in m.idle:
    #     return Constraint.Skip
    return m.idle[s, t] <= 1 - m.charge[s, t]
model.idle_up_charge = Constraint(model.slots, model.times, rule=idle_upper_charge_rule)

def idle_upper_discharge_rule(m, s, t):
    # if (s, t) not in m.idle:
    #     return Constraint.Skip
    return m.idle[s, t] <= 1 - m.discharge[s, t]
model.idle_up_discharge = Constraint(model.slots, model.times, rule=idle_upper_discharge_rule)

def idle_upper_swap_rule(m, s, t):
    # if (s, t) not in m.idle:
    #     return Constraint.Skip
    return m.idle[s, t] <= 1 - m.swap[s, t]
model.idle_up_swap = Constraint(model.slots, model.times, rule=idle_upper_swap_rule)

def idle_lower_rule(m, s, t):
    # if (s, t) not in m.idle:
    #     return Constraint.Skip
    # if all actions are zero -> idle must be 1
    return m.idle[s, t] >= 1 - (m.charge[s, t] + m.discharge[s, t] + m.swap[s, t])
model.idle_low = Constraint(model.slots, model.times, rule=idle_lower_rule)

# 12. Idle Discharged and Below Battery KWH should be Charging
def idle_and_notfull_force_charge_rule(m, s, t):
    # if (s, t) not in m.idle:
    #     return Constraint.Skip
    return m.charge[s, t] >= m.idle[s, t] + m.is_not_full[s, t] - 1
model.idle_and_notfull_force_charge = Constraint(model.slots, model.times, rule=idle_and_notfull_force_charge_rule)

# 13. If model.soc >= BATTERY_KWH/2 (threshold) and there is swap demand, force swap
# Battery ready jika SOC >= threshold
model.has_demand = Param(model.times, initialize=lambda m, t: 1 if swap_hour.get(t, 0) > 0 else 0, within=Binary)

# Kendala 1: Jika baterai siap (battery_ready = 1), maka SOC harus di atas ambang batas.
def soc_ready_if_true_rule(m, s, t):
    return m.soc[s, t] >= threshold - M * (1 - m.battery_ready[s, t])
model.soc_ready_if_true = Constraint(model.slots, model.times, rule=soc_ready_if_true_rule)

# Kendala 2: Jika SOC di atas ambang batas, maka baterai harus siap (battery_ready = 1).
def ready_if_soc_is_sufficient_rule(m, s, t):
    # Logika implikasi: (SOC > threshold) => (battery_ready == 1)
    # Ini memastikan bahwa jika SOC tinggi, maka battery_ready akan bernilai 1.
    return m.battery_ready[s, t] >= (m.soc[s, t] - threshold + eps) / M
model.ready_if_soc_is_sufficient = Constraint(model.slots, model.times, rule=ready_if_soc_is_sufficient_rule)

# Force swap jika ready + demand + tidak discharge

def force_swap_if_ready_and_demand_rule(m, s, t):
    # Jika ready & ada demand & tidak discharge → swap >= 1
    # m.swap adalah biner, jadi swap = 1
    return m.swap[s, t] >= m.battery_ready[s, t] + m.has_demand[t] - 1
model.force_swap_if_ready_and_demand = Constraint(model.slots, model.times, rule=force_swap_if_ready_and_demand_rule)

# # 13. Battery Readiness Constraints
# # 1) Kalau battery_ready = 1 maka SOC >= threshold
# def ready_soc_min(m, s, t):
#     return m.soc[s, t] >= m.threshold - M * (1 - m.battery_ready[s, t])
# model.ready_soc_min = Constraint(model.slots, model.times, rule=ready_soc_min)

# # 2) Kalau battery_ready = 0 maka SOC boleh < threshold
# def ready_soc_max(m, s, t):
#     return m.soc[s, t] <= m.threshold + M * m.battery_ready[s, t]
# model.ready_soc_max = Constraint(model.slots, model.times, rule=ready_soc_max)

# # 3) Ready hanya kalau idle (tidak charge / discharge)
# def ready_if_idle(m, s, t):
#     return m.battery_ready[s, t] <= 1 - m.charge[s, t] - m.discharge[s, t]
# model.ready_if_idle = Constraint(model.slots, model.times, rule=ready_if_idle)

# GIMANA KALO JIKA SEBELUMNYA CHARGING = 1, MAKA MODEL SAAT INI MEMERIKSA APAKAH SEBELUMNYA CHARGING = 1 ATAU TIDAK?
# JIKA YA, MAKA PERIKSA APAKAH MODEL.SOC=1 ATAU TIDAK?
# JIKA YA, PERIKSA APAKAH STATUS SAAT INI DISCHARGING ATAU IDLE?
# JIKA IDLE, MAKA FORCE CHARGE = 1 

# --------------------------
# Solver
# --------------------------
solver = SolverFactory("highs")
# solver.options["mip_rel_gap"] = 0.02
solver.options["time_limit"] = 20  # Limit to 600 seconds (10 minutes)

result = solver.solve(model, tee=True)

# ------------------------------------
# Post-solve per-slot reconstruction and cashflows
# ------------------------------------
# --------------------------
# Post-solve per-slot reporting
# --------------------------

# Berarti klarifikasi simbol:
# ID = Idle Discharged
# D = Discharging
# CM/CG/CS = Charging Mix/Charging Grid/Charging Solar

# Sisanya jika ada swap berarti jadi SID, SCM, SCG, SCS ya

slot_soc = [BATTERY_KWH]*N_SLOTS
available_pv = pv_kw.copy()
rows=[]

for t in HOURS:
    solar_used=0
    grid_used=0
    slot_state={}
    soc_start={}
    soc_end={}
    chg=0 # Charge
    dis=0 # Discharge

    # Assign slot states
    for i in range(N_SLOTS):
        soc_start[i]=round(value(model.soc[i,t-1]) if t>0 else BATTERY_KWH,2) # Sebagai simulasi, anggap soc awal full semua
        soc_now=round(value(model.soc[i,t]),2)
        code = "IF"

        is_swap = value(model.swap[i, t]) >= 0.5
        is_charge = value(model.charge[i, t]) > 0.5
        is_discharge = value(model.discharge[i, t]) > 0.5
        is_full_val = value(model.is_full[i, t]) >= 0.5 

        print(f"Hour: {t}, Slot: {i}, model.swap value: {value(model.swap[i,t])}, is_swap flag: {is_swap}")
        
        if is_discharge:
            code = "D"
        elif is_swap:
            if soc_now >= BATTERY_KWH - 0.01:
                code = "SIF" # Swap then idle full
            else:
                code = "SID" # Swap then idle
        elif is_charge:
            if available_pv[t] > P_CHARGE_MAX: #Seharusnya P Charge Max per slot ya
               code = "CS" # 
               available_pv[t] -= P_CHARGE_MAX
               solar_used += available_pv[t] - P_CHARGE_MAX
            else:
                code = "CG"
                grid_used += P_CHARGE_MAX - available_pv[t]
        else:
            if soc_now >= BATTERY_KWH - 0.01:
                code = "IF"
            else:
                code = "ID"

        # === 🔍 SoC consistency sanity checks ===
        if code == "D" and soc_now > soc_start[i] + 0.01:
            print(f"[!] ERROR at t={t}: Slot{i} is DISCHARGING but SoC increased ({soc_start[i]} → {soc_now})")
        if code in ("CS", "CG", "CM", "SCS", "SCG") and soc_now < soc_start[i] - 0.01:
            print(f"[!] ERROR at t={t}: Slot{i} is CHARGING but SoC decreased ({soc_start[i]} → {soc_now})")

        soc_end[i] = soc_now
        slot_state[i] = code


    # Cashflows
    chg_kWh=P_CHARGE_MAX*sum(value(model.charge[i,t]) for i in model.slots)
    dis_kWh=P_DISCHARGE_MAX*sum(value(model.discharge[i,t]) for i in model.slots)
    grid_bss=max(0,chg_kWh-available_pv[t])
    pln_store_cost=PLN_PRICE*max(0,store_kW[t]-available_pv[t])
    pln_bss_cost=PLN_PRICE*grid_bss
    pln_total=pln_store_cost+pln_bss_cost
    pln_store_revenue=PLN_PRICE*min(store_kW[t],available_pv[t]+value(model.grid_kWh[t]))
    solar_revenue=SOLAR_SELL*min(store_kW[t],available_pv[t])
    swap_revenue=SWAP_PRICE*value(model.swaphit[t])
    net_revenue=pln_store_revenue+solar_revenue+swap_revenue-pln_total

    row={
        "hour":t,
        "swaps":int(value(model.swaphit[t])),
        "unserved":int(value(model.unserved_swap[t])),
        "pv_kW":round(pv_kw[t],2),
        "grid_kWh":round(value(model.grid_kWh[t]),2),
        "total_charge":round(chg_kWh,2),
        "total_discharge":round(dis_kWh,2),
        "PLN_store_cost":round(pln_store_cost,2),
        "PLN_BSS_cost":round(pln_bss_cost,2),
        "PLN_total_cost":round(pln_total,2),
        "PLN_store_revenue":round(pln_store_revenue,2),
        "Solar_store_revenue":round(solar_revenue,2),
        "Swap_revenue":round(swap_revenue,2),
        "Net_profit_hour":round(net_revenue,2)
    }
    for i in range(N_SLOTS):
        row[f"slot{i}_state"]=slot_state[i]
        row[f"slot{i}_soc_start"]=soc_start[i]
        row[f"slot{i}_soc_end"]=soc_end[i]
    rows.append(row)

df=pd.DataFrame(rows)
df.to_csv("./hourly_dispatch_latest.csv",index=False)

profit=value(model.maximize_profit)
total_unserved=sum(int(value(model.unserved_swap[t])) for t in HOURS)

print("\n====== Hourly Summary ======")
print(df.to_string(index=False))
print("===================================")
print(f"Total profit IDR: {profit:,.0f}")
print(f"Unserved swaps: {int(total_unserved)}")
print("CSV written → hourly_dispatch.csv")
print("\n-- Swap status at t=16 --")
for t in range(24):
    print(f"\n--- Hour {t} ---")
    for s in model.slots:
        print(f"Slot {s}: soc={value(model.soc[s,t]):.3f}, is_full={value(model.is_full[s,t])}, swap={value(model.swap[s,t])}")
    print(f"Swaphit = {value(model.swaphit[t])}, Unserved = {value(model.unserved_swap[t])}, Demand = {swap_hour.get(t,0)}")


