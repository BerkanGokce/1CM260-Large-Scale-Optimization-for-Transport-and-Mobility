
class Destroy:

    @staticmethod
    def destroy1(solution, remove_count, random_gen):
        """
        Random Removal.

        Uniformly select customers from all currently served customers
        and remove them from their second-echelon routes.

        Returns the IDs of removed customers.
        """

        removed_customers = []

        # Collect all currently served customer IDs.
        available_customers = [
            customer_ID
            for route in solution.routes_second
            for customer_ID in route.served_load
        ]

        # Randomly select customers without replacement.
        selected_customers = random_gen.sample(
            available_customers,
            min(remove_count, len(available_customers))
        )

        # Remove each selected customer from its route.
        for customer_ID in selected_customers:

            for route in solution.routes_second:

                if customer_ID in route.served_load:

                    customer_location = solution.customers[customer_ID]["location"]

                    route.remove_customer(customer_location)

                    removed_customers.append(customer_ID)
                    break

        # Remove empty second-echelon routes.
        Destroy.remove_empty_routes(solution)

        return removed_customers

    @staticmethod
    def destroy2(solution, remove_count, random_gen):
        pass

    @staticmethod
    def destroy3(solution, remove_count, random_gen):
        pass

    @staticmethod
    def remove_empty_routes(solution):
        """
        Remove second-echelon routes that serve no customers.
        """
        solution.routes_second = [
            route for route in solution.routes_second
            if len(route.locations) > 2
        ]
