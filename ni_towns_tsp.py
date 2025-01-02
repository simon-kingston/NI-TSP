import math
import os
import pickle
import random
import warnings

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from branca.element import MacroElement, Template
from folium.plugins import MiniMap
from geopy.geocoders import Nominatim
import shapely

# Suppress warnings
warnings.filterwarnings("ignore")
ox.settings.requests_kwargs = {"verify": False}
ox.settings.log_console = False

# Define towns and their counties in Northern Ireland
TOWNS_DICT = {
    "Belfast": "County Antrim",  # City
    "Londonderry": "County Londonderry",  # City
    "Craigavon": "County Armagh",  # Large town
    "Newtownabbey": "County Antrim",  # Large town
    "Bangor": "County Down",  # Large town
    "Castlereagh": "County Antrim",  # Large town
    "Lisburn": "County Antrim",  # Large town
    "Ballymena": "County Antrim",  # Large town
    "Newtownards": "County Down",  # Large town
    "Newry": "County Down",  # Large town
    "Carrickfergus": "County Antrim",  # Large town
    "Antrim": "County Antrim",  # Large town
    "Coleraine": "County Londonderry",  # Large town
    "Omagh": "County Tyrone",  # Large town
    "Larne": "County Antrim",  # Large town
    "Lurgan": "County Down",  # Large town
    "Portadown": "County Armagh",  # Large town
    "Banbridge": "County Down",  # Medium town
    "Armagh": "County Armagh",  # Medium town
    "Dungannon": "County Tyrone",  # Medium town
    "Enniskillen": "County Fermanagh",  # Medium town
    "Strabane": "County Tyrone",  # Medium town
    "Cookstown": "County Tyrone",  # Medium town
    "Limavady": "County Londonderry",  # Medium town
    "Downpatrick": "County Down",  # Medium town
    "Ballymoney": "County Antrim",  # Medium town
    "Ballyclare": "County Antrim",  # Medium town
    "Holywood": "County Down",  # Medium town
    "Magherafelt": "County Londonderry",  # Small town
    "Comber": "County Down",  # Small town
    "Warrenpoint": "County Down",  # Small town
    "Newcastle": "County Down",  # Small town
    "Portstewart": "County Londonderry",  # Small town
    "Donaghadee": "County Down",  # Small town
    "Carryduff": "County Down",  # Small town
    "Kilkeel": "County Down",  # Small town
    "Dromore (County Down)": "County Down",  # Small town
    "Greenisland": "County Antrim",  # Small town
    "Ballynahinch": "County Down",  # Small town
    "Coalisland": "County Tyrone",  # Small town
    "Portrush": "County Antrim",  # Small town
    "Ballycastle": "County Antrim",  # Small town
    "Crumlin": "County Antrim",  # Small town
    "Randalstown": "County Antrim",  # Small town
}

# Define file paths for saving/loading data
SHORTEST_ROUTES_FILE = "ni_shortest_routes.pkl"
DISTANCE_MATRIX_FILE = "ni_distance_matrix.pkl"
CACHE_FILE = "ni_road_network.graphml"

# Geocode town coordinates
geolocator = Nominatim(user_agent="NI_TSP")
TOWN_COORDINATES = {}
POINTS = []
for town, county in TOWNS_DICT.items():
    location = geolocator.geocode(f"{town}, Northern Ireland", timeout=10)
    TOWN_COORDINATES[town] = (location.latitude, location.longitude)
    name = str(town)
    lon = round(float(location.longitude), 4)
    lat = round(float(location.latitude), 4)
    POINTS.append({"name": name, "lon": lon, "lat": lat})

# Load or calculate distance matrix and shortest routes
if (
    os.path.exists(SHORTEST_ROUTES_FILE)
    and os.path.exists(DISTANCE_MATRIX_FILE)
    and os.path.exists(CACHE_FILE)
):
    # Load data from files
    print("Loading shortest routes and distance matrix from files...")
    with open(SHORTEST_ROUTES_FILE, "rb") as f:
        shortest_routes = pickle.load(f)
    with open(DISTANCE_MATRIX_FILE, "rb") as f:
        distance_df = pickle.load(f)
    graph = ox.load_graphml(CACHE_FILE)
else:
    # Download road network data for Northern Ireland
    print("Downloading road network data...")
    northern_ireland = ox.geocode_to_gdf("Northern Ireland")
    ulster = ox.geocode_to_gdf("Ulster")
    ni_boundary = (
        northern_ireland.geometry[0].intersection(ulster.geometry[0]).buffer(0.001)
    )
    if isinstance(ni_boundary, shapely.geometry.MultiPolygon):
        ni_polygon = max(ni_boundary.geoms, key=lambda x: x.area)
    else:
        ni_polygon = ni_boundary
    graph = ox.graph_from_polygon(ni_polygon, network_type="drive")

    # Initialize distance matrix and shortest routes
    num_settlements = len(TOWN_COORDINATES)
    distance_matrix = np.full((num_settlements, num_settlements), np.inf)
    shortest_routes = {}

    # Precompute shortest routes and distances
    print("Precomputing shortest routes and distances...")
    for i in range(num_settlements):
        shortest_routes[i] = {}
        for j in range(num_settlements):
            if i == j:
                shortest_routes[i][j] = None
            else:
                town1 = list(TOWN_COORDINATES.keys())[i]
                town2 = list(TOWN_COORDINATES.keys())[j]
                orig_node = ox.nearest_nodes(
                    graph, TOWN_COORDINATES[town1][1], TOWN_COORDINATES[town1][0]
                )
                dest_node = ox.nearest_nodes(
                    graph, TOWN_COORDINATES[town2][1], TOWN_COORDINATES[town2][0]
                )
                try:
                    shortest_routes[i][j] = nx.shortest_path(
                        graph, orig_node, dest_node, weight="length"
                    )
                    shortest_path_length = nx.shortest_path_length(
                        graph, orig_node, dest_node, weight="length"
                    )
                    distance_matrix[i, j] = distance_matrix[j, i] = (
                        shortest_path_length * 0.000621371
                    )
                except nx.NetworkXNoPath:
                    print(f"Warning: No path found between {town1} and {town2}")
                    shortest_routes[i][j] = None

    # Convert distance matrix to Pandas DataFrame
    distance_df = pd.DataFrame(
        distance_matrix, index=TOWN_COORDINATES.keys(), columns=TOWN_COORDINATES.keys()
    )

    # Save calculated data to files
    print("Saving shortest routes and distance matrix to files...")
    with open(SHORTEST_ROUTES_FILE, "wb") as f:
        pickle.dump(shortest_routes, f)
    with open(DISTANCE_MATRIX_FILE, "wb") as f:
        pickle.dump(distance_df, f)
    print(f"Distance matrix saved to {DISTANCE_MATRIX_FILE}")


class GeographicTSP:
    def __init__(self, points, profile):
        if isinstance(points[0], dict):
            self.points = [(p["lon"], p["lat"]) for p in points]
            self.names = [p["name"] for p in points]
        else:
            raise ValueError(
                "Invalid input format. Expected list of (lon, lat) tuples or dictionaries."
            )
        self.length = len(points)
        self.profile = profile

    def solve(self, method="two_opt", **kwargs):
        """Solves the TSP using the specified method."""
        if method == "two_opt":
            self.tour = self.two_opt(starting_route=None)
        elif method == "nearest_neighbor":
            self.tour = self.solve_nearest_neighbor_multiple_starts()
        elif method == "nearest_neighbor_2opt":
            self.tour = self.solve_nearest_neighbor_with_2opt()
        elif method == "genetic":
            self.tour, _ = self.solve_genetic_algorithm(**kwargs)
        elif method == "simulated_annealing":
            self.tour = self.solve_simulated_annealing(**kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")
        if self.tour:
            self.tour.append(self.tour[0])
        return self.tour

    def two_opt(self, starting_route=None):
        """Improves an existing route using the 2-opt algorithm."""
        if starting_route is None:  # If no starting route is provided
            current_route = list(range(len(self.points)))
            random.shuffle(current_route)
        else:
            current_route = starting_route[:]  # Use the provided starting route
        distance_matrix = distance_df.values
        num_towns = len(current_route)
        improved = True
        while improved:
            improved = False
            for i in range(num_towns - 2):
                for j in range(i + 2, num_towns - 1):
                    a, b, c, d = (
                        current_route[i],
                        current_route[i + 1],
                        current_route[j],
                        current_route[(j + 1) % num_towns],
                    )
                    current_distance = distance_matrix[a, b] + distance_matrix[c, d]
                    new_distance = distance_matrix[a, c] + distance_matrix[b, d]
                    if new_distance < current_distance:
                        current_route[i + 1 : j + 1] = current_route[j:i:-1]
                        improved = True
        self.tour = current_route
        return self.tour

    def solve_nearest_neighbor(self):
        """Solves the TSP using the nearest neighbor algorithm."""
        unvisited = set(range(len(self.points)))
        unvisited_list = list(unvisited)
        current_city = random.sample(unvisited_list, 1)[0]
        unvisited.remove(current_city)
        tour = [current_city]
        while unvisited:
            nearest_city = min(
                unvisited, key=lambda city: distance_df.iloc[current_city, city]
            )
            tour.append(nearest_city)
            current_city = nearest_city
            unvisited.remove(nearest_city)
        return tour

    def solve_nearest_neighbor_multiple_starts(self, num_starts=43):
        """Runs nearest neighbor with multiple starting cities."""
        best_tour = None
        best_distance = float("inf")
        for _ in range(num_starts):
            tour = self.solve_nearest_neighbor()
            distance = self.calculate_tour_distance(tour)
            if distance < best_distance:
                best_tour = tour
                best_distance = distance
        return best_tour

    def solve_nearest_neighbor_with_2opt(self):
        """Solves TSP using nearest neighbor followed by 2-opt optimization."""
        tour = self.solve_nearest_neighbor()
        optimized_tour = self.two_opt(
            starting_route=tour
        )  # Pass the nearest neighbor tour to two_opt
        return optimized_tour

    def solve_genetic_algorithm(
        self,
        population_size=50,
        generations=500,
        mutation_rate=0.01,
        elitism=True,
        crossover_method="pmx",
        mutation_method="inversion",
    ):
        """Solves the TSP using a genetic algorithm with an adaptive mutation rate."""

        def create_individual(cities):
            """Creates a random individual (tour)."""
            individual = list(cities)
            random.shuffle(individual)
            return individual

        def calculate_fitness(individual):
            """Calculates the fitness (total distance) of a tour."""
            distance = 0
            for i in range(len(individual) - 1):
                city1 = individual[i]
                city2 = individual[i + 1]
                distance += distance_df.iloc[city1, city2]
            distance += distance_df.iloc[individual[-1], individual[0]]
            return distance

        def crossover_pmx(parent1, parent2):
            """Performs Partially Mapped Crossover (PMX)."""
            size = len(parent1)
            p1, p2 = [0] * size, [0] * size
            # Initialize the position of each index in the individuals
            for i in range(size):
                p1[parent1[i]] = i
                p2[parent2[i]] = i
            # Choose crossover points
            cxpoint1 = random.randint(0, size)
            cxpoint2 = random.randint(0, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1
            # Apply crossover between cx points
            for i in range(cxpoint1, cxpoint2):
                # Keep track of the selected values
                temp1 = parent1[i]
                temp2 = parent2[i]
                # Swap the matched value
                parent1[i], parent1[p1[temp2]] = temp2, temp1
                parent2[i], parent2[p2[temp1]] = temp1, temp2
                # Position bookkeeping
                p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
                p2[temp1], p2[temp2] = p2[temp2], p2[temp1]
            return parent1, parent2

        def crossover_ox(parent1, parent2):
            """Performs Order Crossover (OX)."""
            size = len(parent1)
            child1 = [-1] * size
            child2 = [-1] * size
            # Choose random slice points
            start, end = sorted(random.sample(range(size), 2))
            # Copy the slice from parent1 to child1
            child1[start:end] = parent1[start:end]
            # Fill the rest of child1 with genes from parent2, maintaining order
            j = end
            for i in range(end, size):
                while parent2[j % size] in child1:
                    j += 1
                child1[i] = parent2[j % size]
                j += 1
            for i in range(start):
                while parent2[j % size] in child1:
                    j += 1
                child1[i] = parent2[j % size]
                j += 1
            # Copy the slice from parent2 to child2
            child2[start:end] = parent2[start:end]
            # Fill the rest of child2 with genes from parent1, maintaining order
            j = end
            for i in range(end, size):
                while parent1[j % size] in child2:
                    j += 1
                child2[i] = parent1[j % size]
                j += 1
            for i in range(start):
                while parent1[j % size] in child2:
                    j += 1
                child2[i] = parent1[j % size]
                j += 1
            return child1, child2

        def mutate_inversion(individual, mutation_rate):
            """Performs inversion mutation."""
            if random.random() < mutation_rate:
                start, end = sorted(random.sample(range(len(individual)), 2))
                individual[start:end] = individual[start:end][::-1]
            return individual

        def mutate_insertion(individual, mutation_rate):
            """Performs insertion mutation."""
            if random.random() < mutation_rate:
                from_index = random.randint(0, len(individual) - 1)
                to_index = random.randint(0, len(individual) - 1)
                city = individual.pop(from_index)
                individual.insert(to_index, city)
            return individual

        # Error handling for crossover and mutation methods
        valid_crossover_methods = ["pmx", "ox"]
        valid_mutation_methods = ["inversion", "insertion"]
        if crossover_method not in valid_crossover_methods:
            raise ValueError(
                f"Invalid crossover method: {crossover_method}. Valid methods are: {valid_crossover_methods}"
            )
        if mutation_method not in valid_mutation_methods:
            raise ValueError(
                f"Invalid mutation method: {mutation_method}. Valid methods are: {valid_mutation_methods}"
            )

        # Select crossover and mutation methods
        crossover_func = crossover_pmx if crossover_method == "pmx" else crossover_ox
        mutate_func = (
            mutate_inversion if mutation_method == "inversion" else mutate_insertion
        )

        population = [
            create_individual(range(len(self.points))) for _ in range(population_size)
        ]
        fitness_history = []  # Store the best fitness in each generation

        for generation in range(generations):
            fitness_scores = [
                calculate_fitness(individual) for individual in population
            ]
            fitness_history.append(min(fitness_scores))  # Record best fitness

            # Adaptive Mutation Rate (decrease every 100 generations)
            if generation % 100 == 0 and generation > 0:
                mutation_rate *= 0.9

            # Elitism
            elite_individuals = []
            if elitism:
                elite_individuals = sorted(population, key=calculate_fitness)[:2]

            parents = []
            for _ in range(max(2, population_size - len(elite_individuals))):
                tournament_indices = random.sample(range(len(population)), 5)
                winner_index = min(tournament_indices, key=lambda i: fitness_scores[i])
                parents.append(population[winner_index])

            offspring = []
            for i in range(0, len(parents) - 1, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child1, child2 = crossover_func(parent1, parent2)
                child1 = mutate_func(child1, mutation_rate)
                child2 = mutate_func(child2, mutation_rate)
                offspring.extend([child1, child2])

            population = offspring + elite_individuals if elitism else offspring

        best_individual = min(population, key=calculate_fitness)
        self.tour = best_individual
        return self.tour, fitness_history

    def solve_simulated_annealing(
        self, starting_temperature=10000, cooling_rate=0.9995
    ):  # Adjusted parameters
        """Solves the TSP using Simulated Annealing."""
        current_route = list(range(len(self.points)))
        random.shuffle(current_route)
        current_distance = self.calculate_tour_distance(current_route)
        temperature = starting_temperature
        while temperature > 1:
            i, j = sorted(random.sample(range(len(self.points)), 2))
            neighbor_route = current_route[:]
            neighbor_route[i:j] = neighbor_route[j - 1 : i - 1 : -1]
            expected_cities = set(range(len(self.points)))
            neighbor_cities = set(neighbor_route)
            if neighbor_cities != expected_cities:
                continue
            neighbor_distance = self.calculate_tour_distance(neighbor_route)
            delta_distance = neighbor_distance - current_distance
            acceptance_probability = math.exp(-delta_distance / temperature)
            if delta_distance < 0 or random.random() < acceptance_probability:
                current_route = neighbor_route[:]
                current_distance = neighbor_distance
            temperature *= cooling_rate
        self.tour = current_route[:]
        return self.tour

    def calculate_tour_distance(self, route):
        """Calculates the total distance of a tour."""
        if len(route) < 2:
            return 1000000000000
        distance = 0
        for i in range(len(route) - 1):
            city1 = route[i]
            city2 = route[i + 1]
            distance += distance_df.iloc[city1, city2]
        distance += distance_df.iloc[route[-1], route[0]]
        return distance

    def get_directions(self):
        """Calculates route information using the precomputed distance matrix."""
        self.total_distance = 0
        self.route_segments = []
        for i in range(len(self.tour)):
            start_idx = self.tour[i]
            end_idx = self.tour[(i + 1) % len(self.tour)]
            start_point = self.points[start_idx]
            end_point = self.points[end_idx]
            route = shortest_routes[start_idx if start_idx == 0 else start_idx - 1][
                end_idx if end_idx == 0 else end_idx - 1
            ]
            start_town = self.names[start_idx]
            end_town = self.names[end_idx]
            distance = distance_df.loc[start_town, end_town]
            if np.isinf(distance):
                distance = 0
            self.total_distance += distance
            self.route_segments.append(
                {
                    "start": start_point,
                    "end": end_point,
                    "distance": distance,
                    "route": route,
                }
            )
        return self.route_segments

    def generate_map(self, graph, route=None):
        """Generates a Folium map of the solution."""
        route_points = [(point[1], point[0]) for point in self.points]
        center = np.mean([x for (x, y) in route_points]), np.mean(
            [y for (x, y) in route_points]
        )
        m = folium.Map(location=center, zoom_start=9, zoom_control=False)

        # setting up a minimap for general orientation when on zoom
        miniMap = MiniMap(
            toggle_display=True,
            zoom_level_offset=-5,
            tile_layer="cartodbdark_matter",
            width=140,
            height=100,
            minimized=False,
        ).add_to(m)
        m.add_child(miniMap)

        route_line_group = folium.FeatureGroup(
            name="Route Line", show=True, control=False
        )
        if route is not None:
            if route:
                route = route[:-1]  # Drop the last element
            # Rotate the tour until "Belfast" is the first element
            while self.names[route[0]] != "Belfast":
                route = route[1:] + [route[0]]
            route.append(route[0])

            total_distance = 0
            for i in range(len(route) - 1):
                town1_index = route[i]
                town2_index = route[(i + 1) % len(route)]
                town1 = self.names[town1_index]
                town2 = self.names[town2_index]
                distance_to_next = distance_df.loc[town1, town2]
                total_distance += distance_to_next
                node1 = ox.nearest_nodes(
                    graph, TOWN_COORDINATES[town1][1], TOWN_COORDINATES[town1][0]
                )
                node2 = ox.nearest_nodes(
                    graph, TOWN_COORDINATES[town2][1], TOWN_COORDINATES[town2][0]
                )
                route_nodes = nx.shortest_path(graph, node1, node2, weight="length")

                # Handle direct edge case for route coordinates
                if len(route_nodes) == 2:
                    route_coordinates = [
                        (
                            graph.nodes[route_nodes[0]]["y"],
                            graph.nodes[route_nodes[0]]["x"],
                        ),
                        (
                            graph.nodes[route_nodes[1]]["y"],
                            graph.nodes[route_nodes[1]]["x"],
                        ),
                    ]
                else:
                    route_coordinates = [
                        (graph.nodes[node]["y"], graph.nodes[node]["x"])
                        for node in route_nodes
                    ]
                folium.PolyLine(
                    route_coordinates, color="red", weight=2, opacity=0.5
                ).add_to(route_line_group)
        route_line_group.add_to(m)

        route_points_group = folium.FeatureGroup(
            name="Route Points",
            show=True,
            control=False,
        )

        # Create the table rows dynamically
        table_rows = ""
        cumulative_distance = 0
        for i in range(len(route) - 1):
            town1_index = route[i]
            town2_index = route[(i + 1) % len(route)]
            town1 = self.names[town1_index]
            town2 = self.names[town2_index]
            distance_to_next = distance_df.loc[town1, town2]
            cumulative_distance += distance_to_next
            table_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td>{town1}</td>
                <td>{town2}</td>
                <td>{distance_to_next:.2f}</td>
                <td>{cumulative_distance:.2f}</td>
            </tr>
            """

        legend_html = f"""
        {{% macro html(this, kwargs) %}}
        <!doctype html>
        <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>NI TSP</title>
                <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.1/css/all.min.css" integrity="sha512-MV7K8+y+gLIBoVD59lQIYicR65iaqukzvf/nwasF0nqhPay5w/9lJmVM2hMDcnK1OnMGCdVK+iQrJ7lzPJQd1w==" crossorigin="anonymous" referrerpolicy="no-referrer"/>
                <link rel="stylesheet" href="src/ui.css">
                <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
                <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
                <script>
                    $(function() {{
                        $("#ui-container, #title-container, #project-container").draggable({{
                            start: function(event, ui) {{
                                $(this).css({{
                                    right: "auto",
                                    top: "auto",
                                    bottom: "auto"
                                }});
                            }}
                        }});
                    }});
                </script>
            </head>
            <body>
            <div class="ui-container" id="title-container">
                <div class="map-title">
                <p>Northern Ireland Travelling Salesman Problem</p>
                </div>
            </div>
            <div id="ui-container" class="ui-container">
                <div class="index-container">
                    <div class='legend-scale'>
                        <table style="width: auto; font-size: 12px; line-height: 1.2;">
                            <thead>
                                <tr>
                                    <th>Stop</th>
                                    <th>From</th>
                                    <th>To</th>
                                    <th>Distance<br>(miles)</th>
                                    <th>Cumulative<br>Distance<br>(miles)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {table_rows}
                            </tbody>
                        </table>
                        <div class="leaflet-control-layers-separator"></div>
                        <p style="font-weight: bold; text-decoration: underline double;">Total Distance: {total_distance:.2f} miles</p> 
                    </div>
                </div>
            </div>
            </body>
        </html>
        {{% endmacro %}}
        """
        # Add the legend using a MacroElement
        legend = MacroElement()
        legend._template = Template(legend_html)
        legend._template.render(table_rows=table_rows)
        m.get_root().add_child(legend)

        total_distance = 0
        for i in range(len(route) - 1):
            town_index = route[i]
            current_node = self.names[town_index]
            popup_text = f"<strong>{i}: {current_node}</strong>"
            next_town_index = route[(i + 1) % len(route)]
            next_town = self.names[next_town_index]
            distance_to_next = distance_df.loc[current_node, next_town]
            total_distance += distance_to_next
            folium.CircleMarker(
                location=(self.points[town_index][1], self.points[town_index][0]),
                radius=6,
                color="white",
                fill=True,
                fill_color="blue",
                fill_opacity=1,
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=popup_text,
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10pt;color:white;font-weight:bold;">{i}</div>'
                ),
                size=2,
            ).add_to(route_points_group)
        route_points_group.add_to(m)

        # Add the custom tile layer to the map
        custom_tile_layer = folium.TileLayer(
            tiles="http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            attr="CartoDB Positron",
            name="Positron",
            overlay=True,
            control=False,
            opacity=0.7,
            show=True,
        )
        custom_tile_layer.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        return m


# Create GeographicTSP object
ni_towns = GeographicTSP(points=POINTS, profile="car")

# Define a dictionary to store results
results = {}

# Store the best tour found so far
best_tour = None
best_distance = float("inf")

# Define hyperparameter sets for each method
hyperparameter_sets = {
    "two_opt": [{}],
    "nearest_neighbor": [{}],
    "nearest_neighbor_2opt": [{}] * 30,  # Run 30 times
    "genetic": [
        {
            "population_size": 50,
            "generations": 750,
            "mutation_rate": 0.01,
            "elitism": True,
            "crossover_method": "ox",
            "mutation_method": "inversion",
        }
    ],
    "simulated_annealing": [
        {"starting_temperature": 18000, "cooling_rate": 0.9997},
    ],
}

for method, parameter_sets in hyperparameter_sets.items():
    for i, parameters in enumerate(parameter_sets):
        initial_route = list(range(len(POINTS)))
        initial_route.append(initial_route[0])
        initial_distance = sum(
            distance_df.loc[
                ni_towns.names[initial_route[i]], ni_towns.names[initial_route[i + 1]]
            ]
            for i in range(len(initial_route) - 1)
        )

        tour = ni_towns.solve(method=method, **parameters)  # Pass parameters to solve()
        ni_towns.get_directions()

        # --- Check if reversed tour is shorter ---
        reversed_tour = ni_towns.tour[::-1]  # Reverse the tour
        reversed_distance = ni_towns.calculate_tour_distance(reversed_tour)
        if reversed_distance < ni_towns.total_distance:
            print(f"Reversed tour is shorter: {reversed_distance:.2f} miles")
            ni_towns.tour = reversed_tour[:]  # Update the tour in the ni_towns object
            ni_towns.total_distance = reversed_distance  # Update the total distance

        results[f"{method}_{i+1}"] = {  # Unique key for each parameter set
            "initial_tour": initial_route,
            "optimized_tour": tour,
            "initial_distance": initial_distance,
            "total_distance": ni_towns.total_distance,
            "parameters": parameters,
        }

        print(f"--- {method.upper()} ({i+1}) ---")
        print(f"Tour: {results[f'{method}_{i+1}']['optimized_tour']}")
        print(
            f"Total Tour Length: {results[f'{method}_{i+1}']['total_distance']:.2f} miles"
        )

# Find the best result
best_method_key = min(results, key=lambda k: results[k]["total_distance"])

# Generate map only for the best route
ni_towns.generate_map(graph, route=results[best_method_key]["optimized_tour"]).save(
    f"ni_towns_tsp.html"
)

# Print a summary of the results
print("\n--- RESULTS SUMMARY ---")
sorted_results = sorted(results.items(), key=lambda item: item[1]["total_distance"])
for method_key, data in sorted_results:
    print(
        f"{method_key.upper()}: Shortest distance {(data['total_distance']):.2f} miles (Parameters: {data['parameters']})"
    )
