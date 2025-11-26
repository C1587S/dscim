import xarray as xr
from dscim.menu.main_recipe import MainRecipe


class Baseline(MainRecipe):
    """Adding up option"""

    NAME = "adding_up"
    __doc__ = MainRecipe.__doc__

    def ce_cc(self):
        pass

    def ce_no_cc(self):
        pass

    def global_damages_calculation(self):
        """Call global damages"""
        return self.adding_up_damages.to_dataframe("damages").reset_index()

    def damages_calculation(self, geography) -> xr.Dataset:
        """Aggregate damages to country/region level

        Returns
        --------
            xr.Dataset with damages variable
        """
        self.logger.info(f"Calculating damages for geography: {geography}")

        # Get damages from adding_up (returns as xarray)
        dams_collapse = self.adding_up_damages

        if geography == "ir":
            pass
        elif geography == "country":
            territories = []
            mapping_dict = {}
            for ii, row in self.countries_mapping.iterrows():
                mapping_dict[row["ISO"]] = row["MatchedISO"]
                if row["MatchedISO"] == "nan":
                    mapping_dict[row["ISO"]] = "nopop"

            for region in dams_collapse.region.values:
                    territories.append(mapping_dict[region[:3]])

            dams_collapse = (dams_collapse
                             .assign_coords({'region':territories})
                             .groupby('region')
                             .sum())
        elif geography == "globe":
            dams_collapse = dams_collapse.sum(dim="region").assign_coords({'region':'globe'}).expand_dims('region')

        if "gwr" in self.discounting_type:
            dams_collapse = dams_collapse.assign(
                ssp=str(list(self.gdp.ssp.values)),
                model=str(list(self.gdp.model.values)),
            )

        return dams_collapse.to_dataset(name = 'damages')

    def calculated_damages(self):
        pass

    def ce_cc_calculation(self):
        pass

    def ce_test(self):
        pass

    def ce_no_cc_calculation(self):
        pass

    def global_consumption_calculation(self, disc_type):
        """Calculate global consumption"""

        if (disc_type == "constant") or ("ramsey" in disc_type):
            global_cons_no_cc = self.gdp.sum(dim=["region"])

        elif disc_type == "constant_model_collapsed":
            global_cons_no_cc = self.gdp.sum(dim=["region"]).mean(dim=["model"])

        elif "gwr" in disc_type:
            global_cons_no_cc = self.gdp.sum(dim=["region"]).mean(dim=["model", "ssp"])

        global_cons_no_cc.name = f"global_cons_{disc_type}"

        return global_cons_no_cc
