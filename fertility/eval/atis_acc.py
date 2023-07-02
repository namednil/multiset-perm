from typing import Dict

from fertility.eval.funql_acc import FunqlAcc
from fertility.eval.lev import MyMetric


atis_arities = dict({('_stop_1', 1), ('argmax_capacity', 1), ('_fare_2', 1), ('_year_2', 1), ('year', 1), ('_to_2', 1), ('_economy', 1),
                ('_named_1', 1), ('days_code', 1), ('_rapid_transit', 1), ('_oneway', 1), ('month', 1), ('meal_description', 1),
                ('_<_departure_time_2', 1), ('_turboprop', 1), ('_booking_class_2', 1), ('_flight_aircraft', 1),
                ('_stops', 1), ('argmax_count', 1), ('_booking_class_1', 1), ('_booking_class:_t', 1), ('manufacturer', 1),
                ('_flight_number_2', 1), ('_round_trip', 1), ('_jet', 1), ('_fare_time', 1), ('fare_basis_code', 1), ('argmin_fare', 1),
                ('_tomorrow', 1), ('_flight_airline', 1), ('answer', 1), ('_minimum_connection_time', 1), ('_next_days_2', 1),
                ('_aircraft_basis_type_2', 1), ('argmin_miles_distant_2', 1), ('_limousine', 1), ('_fare', 1), ('_arrival_time', 1),
                ('_stop_2', 1), ('_nonstop', 1), ('_daily', 1), ('_aircraft_1', 1), ('_aircraft', 1), ('_services', 2), ('class_description', 1),
                ('argmax_stops', 1), ('_flight', 1), ('_stop_arrival_time', 1), ('argmin_time_elapsed', 1), ('day_period', 1), ('_from_2', 1),
                ('_loc:_t_1', 1), ('_time_zone_code', 1), ('_airport_1', 1), ('city_name', 1), ('time', 1), ('_time_elapsed', 1),
                ('_connecting', 1), ('_fare_basis_code', 1), ('_approx_return_time_2', 1), ('_meal_code_2', 1),
                ('day_number', 1), ('argmin_stops', 1), ('argmin_departure_time', 1), ('_from_1', 1),
                ('_airline_name', 1), ('_day_after_tomorrow', 1), ('_city', 1),
                ('_airport', 1), ('_month_2', 1), ('_ground_transport', 1), ('argmax_fare', 1),
                ('_restriction_code', 1), ('_to_1', 1), ('_meal', 1), ('_has_stops', 1),
                ('_>_arrival_time_2', 1), ('_<_fare_2', 1), ('_abbrev', 1), ('_days_from_today_2', 1),
                ('_overnight', 1), ('_during_day_2', 1), ('_airline', 1), ('airport_code', 1), ('_meal_code', 1),
                ('airline_code', 1), ('_after_day_2', 1), ('sum_stops', 1), ('_discounted', 1), ('_today', 1),
                ('_before_day_2', 1), ('_month_return_2', 1), ('_class_type_2', 1), ('_flight_number', 1),
                ('sum_capacity', 1), ('_day_number_return_2', 1), ('_min', 1), ('_time_elapsed_2', 1),
                ('_class_of_service', 1), ('_rental_car', 1), ('_max', 1), ('_fare_basis_code_2', 1),
                ('_aircraft_2', 1), ('_miles_distant', 2), ('_from_airport_2', 1),
                ('_air_taxi_operation', 1), ('_tomorrow_arrival', 1), ('_day_arrival_2', 1), ('_<_arrival_time_2', 1),
                ('aircraft_code', 1), ('_loc:_t_2', 1), ('flight_number', 1), ('_services_1', 1), ('count', 1),
                ('_airline_1', 1), ('_day_2', 1), ('_meal_2', 1), ('_to_city_2', 1), ('_capacity', 1),
                ('argmax_arrival_time', 1), ('_departure_time', 1), ('_taxi', 1), ('argmax_departure_time', 1),
                ('_during_day_arrival_2', 1), ('_approx_arrival_time_2', 1), ('_day_return_2', 1), ('_>_stops_2', 1),
                ('_manufacturer_2', 1), ('state_name', 1), ('day', 1), ('basis_type', 1), ('_departure_time_2', 1),
                ('_tonight', 1), ('_weekday', 1), ('_has_meal', 1), ('_day_number_arrival_2', 1), ('dollar', 1),
                ('_>_capacity_2', 1), ('argmin_capacity', 1), ('argmin_arrival_time', 1), ('_ground_fare', 1),
                ('hour', 1), ('_flight_fare', 1), ('not', 1), ('_services_2', 1), ('integer', 1),
                ('_month_arrival_2', 1), ('_equals', 2), ('_airline_2', 1), ('_day_number_2', 1),
                ('meal_code', 1), ('_minutes_distant', 1), ('_stops_2', 1), ('_>_departure_time_2', 1),
                ('_arrival_time_2', 1), ('_approx_departure_time_2', 1)})


@MyMetric.register("atis_acc")
class AtisAcc(FunqlAcc):

    def __init__(self):
        super().__init__(atis_arities, sortable_nodes = ("intersection", "or"))





