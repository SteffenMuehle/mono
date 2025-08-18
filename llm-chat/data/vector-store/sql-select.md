select
    trip_id,
    fleetid as fleet_id,
    activity as gandalfs_activity
from table1
where
    ts > '2025'