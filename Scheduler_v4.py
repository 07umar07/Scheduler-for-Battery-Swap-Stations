import pandas as pd
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

# Maximum Power of Charging
P_CHARGE_MAX = 0.8

# Maximum Power of disharging
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
LOAD_OPEN_KW = 8.2

# Store consumption (kW) during closed hours
LOAD_CLOSED_KW = 2.3

# Store open hours
OPEN_HOURS = range(7, 24)

# PV Panel Parameters
PV_PANEL_WP = 300    # watts-peak per panel
PV_PANEL_COUNT = 9   # number of panels

# Irradiance per hour (W/m2)
IRRADIANCE = {
    7:114, 8:343, 9:554, 10:717, 11:820, 12:854,
    13:810, 14:618, 15:337, 16:211, 17:101
}

# Recorded swap timestamps
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

# If Swap success in given slot and time, it called swaphit btw
model.swaphit = Var(model.times, domain = NonNegativeIntegers, bounds= (0, N_SLOTS))

# If Swap failed in given slot and time, it called unserved swap
model.unserved_swap = Var(model.times, domain = NonNegativeIntegers, bounds= (0, N_SLOTS))

# Power of Grid which could be drawn for either discharing or charging
model.grid_kWh = Var(model.times, domain = NonNegativeReals)

# Excessive Solar Power (Not swapped nor discharged)
model.excess_solar = Var(model.times, domain = NonNegativeReals)

# ---------------
# Objective
# ---------------

# Revenue for all swaps that happened
revenue_swap = SWAP_PRICE * sum(model.swaphit[t] for t in HOURS)

# Revenue from Excessed Solar 
revenue_solar = SOLAR_SELL * sum(max(0, pv_kw[t] - store_kW[t]) for t in HOURS)

# Cost to buy electricity from grid to discharged to store
cost_grid = PLN_PRICE * sum(model.grid_kWh[t] for t in HOURS)

# Cost from battery degradation due to full swaps and discharges
cost_degr = DEGR_COST * (
    BATTERY_KWH * sum(model.swap[i, t] for i in model.slots for t in model.times) + 
    P_DISCHARGE_MAX * sum(model.discharge[i, t] for i in model.slots for t in model.times)
)

# Penalty for unserved swaps
penalty_unserved = 100000 * sum(model.unserved_swap[t] for t in HOURS)

# Model Objective
model.maximize_profit = Objective (expr = revenue_swap + revenue_solar - cost_grid - cost_degr - penalty_unserved,
sense = maximize
)

# --------------------------
# Constraint
# --------------------------

# 1. Exclusivity, where only one of charge/discharge/swap could happen at the time (charge + discharge + swap <= 1)
def exclusive_actions_rule(models, slots, times):
    return model.charge[slots, times] + model.discharge[slots, times] + model.swap[slots, times] <= 1
model.exclusive_actions = Constraint(model.slots, model.times, rule= exclusive_actions_rule)

# 2. SOC = SOC + Charging Power * Charge - Discharging Power * Discharge - Energy Loss due to Swap
def soc_dynamic_rule(model, slots, times):
    if times == model.times.first():
        return Constraint.Skip  # Initial SoC handled separately

    time_previous = model.times.prev(times) # Previous TIme

    return model.soc[slots, times] == (
        model.soc[slots, time_previous] +
        P_CHARGE_MAX * model.charge[slots, times] -
        P_DISCHARGE_MAX * model.discharge[slots, times] -
        BATTERY_KWH * model.swap[slots, times]
    )

model.dyn_rule = Constraint(model.slots, model.times, rule=soc_dynamic_rule)

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
    
    time_previous = model.times.prev(times) # Previous TIme
    M = BATTERY_KWH
    epsilon = 0.01

    return model.soc[slots, times] <= BATTERY_KWH - epsilon + M * (1 - model.needs_charge[slots, times])
model.needs_charge_from_soc = Constraint(model.slots, model.times, rule=need_charge_soc_rule)

# 4. Atleast 2 slots ready for swapping
model.is_full = Var(model.slots, model.times, domain=Binary)

def is_full_logic_rule(model, slots, times):
    M = BATTERY_KWH
    epsilon = 0.1  # margin under full capacity
    return model.soc[slots, times] >= BATTERY_KWH - epsilon - M * (1 - model.is_full[slots, times])
model.full_logic = Constraint(model.slots, model.times, rule=is_full_logic_rule) # This forces model.is_full[i, t] = 1 only when soc[i, t] ≥ BATTERY_KWH - epsilon

def minimum_ready_packs_rule(model, times):
    return sum(model.is_full[i, times] for i in model.slots) >= 2
model.min_ready_packs = Constraint(model.times, rule=minimum_ready_packs_rule)

# 5. Total swap hit and swap unserved in a given time shouldnt more than all swap happened in the given time
def swap_fulfillment_rule(model, times):
    return model.swaphit[times] + model.unserved_swap[times] == swap_hour.get(times, 0)
model.swap_fulfillment = Constraint(model.times, rule= swap_fulfillment_rule) 

# 6. Energy Balance: Solar + Discharge + Grid >= Store Load + Battery Charging
def energy_balance_rule(model, times):

    discharge_sum = sum(model.discharge[i, times] for i in model.slots)
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

# --------------------------
# Solver
# --------------------------
solver = SolverFactory("highs")
solver.options["mip_rel_gap"] = 0.02
# solver.options["time_limit"] = 20  # Limit to 600 seconds (10 minutes)

result = solver.solve(model, tee=True)

# ------------------------------------
# Post-solve per-slot reconstruction and cashflows
# ------------------------------------
# --------------------------
# Post-solve per-slot reporting
# --------------------------
slot_soc = [BATTERY_KWH]*N_SLOTS
available_pv = pv_kw.copy()
rows=[]

for t in HOURS:
    solar_used=0
    grid_used=0
    slot_state={}
    soc_start={}
    soc_end={}
    chg=0
    dis=0

    # Assign slot states
    for i in range(N_SLOTS):
        soc_start[i]=round(value(model.soc[i,t-1]) if t>0 else BATTERY_KWH,2)
        soc_now=round(value(model.soc[i,t]),2)
        code = "ID"

        is_swap = value(model.swap[i, t]) >= 0.5
        is_charge = value(model.charge[i, t]) > 0.5
        is_discharge = value(model.discharge[i, t]) > 0.5

        if is_swap:
            if available_pv[t] >= P_CHARGE_MAX:
                code = "SCS"
                solar_used += P_CHARGE_MAX
                available_pv[t] -= P_CHARGE_MAX
            else:
                code = "SCG"
        elif is_charge:
            if available_pv[t] >= P_CHARGE_MAX:
                code = "CS"
                available_pv[t] -= P_CHARGE_MAX 
                solar_used += P_CHARGE_MAX
            elif 0 < available_pv[t] < P_CHARGE_MAX:
                code = "CM"
                solar_used += available_pv[t] 
                available_pv[t] = 0
            else:
                code = "CG"
        elif is_discharge:
            if soc_now > 0.1:
                code = "D"
            else:
                code = "ID"
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
df.to_csv("hourly_dispatch_updated_3.csv",index=False)

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
