import logging
import time
import os
from copy import deepcopy
from typing import Dict, Callable, Tuple, Union, List, Optional

from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
from sqlalchemy import create_engine

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets.sample_dataset import SampleEHRDataset
from pyhealth.datasets.utils import MODULE_CACHE_PATH, DATASET_BASIC_TABLES
from pyhealth.datasets.utils import hash_str, strptime
from pyhealth.medcode import CrossMap
from pyhealth.utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)
 
INFO_MSG = """
dataset.patients: patient_id -> <Patient>

<Patient>
    - visits: visit_id -> <Visit> 
    - other patient-level info
    
    <Visit>
        - event_list_dict: table_name -> List[Event]
        - other visit-level info
    
        <Event>
            - code: str
            - other event-level info
"""


class CDMDataset():
    """Base dataset for OMOP dataset.

    It can then be converted to samples dataset for different tasks by calling `self.set_task()`.

    The Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM)
    is an open community data standard, designed to standardize the structure
    and content of observational data and to enable efficient analyses that
    can produce reliable evidence.

    See: https://www.ohdsi.org/data-standardization/the-common-data-model/.

    The basic information is stored in the following tables:
        - person: contains records that uniquely identify each person or patient,
            and some demographic information.
        - visit_occurrence: contains info for how a patient engages with the
            healthcare system for a duration of time.
        - death: contains info for how and when a patient dies.

    We further support the following tables:
        - condition_occurrence.csv: contains the condition information
            (CONDITION_CONCEPT_ID code) of patients' visits.
        - procedure_occurrence.csv: contains the procedure information
            (PROCEDURE_CONCEPT_ID code) of patients' visits.
        - drug_exposure.csv: contains the drug information (DRUG_CONCEPT_ID code)
            of patients' visits.
        - measurement.csv: contains all laboratory measurements
            (MEASUREMENT_CONCEPT_ID code) of patients' visits.

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]). Basic tables will be loaded by default.
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                - a str of the target code vocabulary. E.g., {"NDC", "ATC"}.
                - a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method. E.g.,
                    {"NDC", ("ATC", {"target_kwargs": {"level": 3}})}.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.
        sep: the separator used for parsing csv files. Default is tab.
        use_compressed: whether to use .csv.gz files. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> dataset = OMOPDataset(
        ...         root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        ...         tables=["condition_occurrence", "procedure_occurrence", "drug_exposure", "measurement",],
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(
        self,
        root: Optional[str] = None,
        database_url: Optional[str] = None,
        tables: List[str] = None,
        dataset_name: Optional[str] = None,
        code_mapping: Optional[Dict[str, Union[str, Tuple[str, Dict]]]] = None,
        dev: bool = False,
        refresh_cache: bool = False,
        sep: str = "\t",
        use_compressed: str = False,
    ):
        """Loads tables into a dict of patients and saves it to cache."""

        if code_mapping is None:
            code_mapping = {}

        # base attributes
        self.dataset_name = (
            self.__class__.__name__ if dataset_name is None else dataset_name
        )
        
        assert root is not None or database_url is not None, "Either root or database_url must be passed in."
        self.root = root
        if database_url is not None:
            self.engine = create_engine(database_url)

        self.code_mapping = code_mapping
        self.dev = dev
        self.sep = sep
        self.use_compressed = use_compressed

        # if we are using a premade dataset, no basic tables need to be provided.
        if self.dataset_name in DATASET_BASIC_TABLES and [
            table
            for table in tables
            if table in DATASET_BASIC_TABLES[self.dataset_name]
        ]:
            raise AttributeError(
                f"Basic tables are parsed by default and do not need to be explicitly selected. Basic tables for {self.dataset_name}: {DATASET_BASIC_TABLES[self.dataset_name]}"
            )

        self.tables = tables

        # the medcode vocabularies of the dataset
        self.code_vocs = {}
        # load medcode for code mapping
        self.code_mapping_tools = self._load_code_mapping_tools()

        # hash filename for cache
        args_to_hash = (
            [self.dataset_name, root]
            + sorted(tables)
            + sorted(code_mapping.items())
            + ["dev" if dev else "prod"]
        )
        filename = hash_str("+".join([str(arg) for arg in args_to_hash])) + ".pkl"
        self.filepath = os.path.join(MODULE_CACHE_PATH, filename)
        
        # check if cache exists or refresh_cache is True
        if os.path.exists(self.filepath) and (not refresh_cache):
            # load from cache
            logger.debug(
                f"Loaded {self.dataset_name} base dataset from {self.filepath}"
            )
            try:
                self.patients, self.code_vocs = load_pickle(self.filepath)
            except:
                raise ValueError("Please refresh your cache by set refresh_cache=True")
        
        else:
            # load from raw data
            logger.debug(f"Processing {self.dataset_name} base dataset...")
            # parse tables
            patients = self.parse_tables()

            # convert codes
            patients = self._convert_code_in_patient_dict(patients)
            self.patients = patients
            # save to cache
            logger.debug(f"Saved {self.dataset_name} base dataset to {self.filepath}")
            save_pickle((self.patients, self.code_vocs), self.filepath)

    def _load_code_mapping_tools(self) -> Dict[str, CrossMap]:
        """Helper function which loads code mapping tools CrossMap for code mapping.

        Will be called in `self.__init__()`.

        Returns:
            A dict whose key is the source and target code vocabulary and
                value is the `CrossMap` object.
        """
        code_mapping_tools = {}
        for s_vocab, target in self.code_mapping.items():
            if isinstance(target, tuple):
                assert len(target) == 2
                assert type(target[0]) == str
                assert type(target[1]) == dict
                assert target[1].keys() <= {"source_kwargs", "target_kwargs"}
                t_vocab = target[0]
            else:
                t_vocab = target
            # load code mapping from source to target
            code_mapping_tools[f"{s_vocab}_{t_vocab}"] = CrossMap(s_vocab, t_vocab)
        return code_mapping_tools

    def parse_tables(self) -> Dict[str, Patient]:
        """Parses the tables in `self.tables` and return a dict of patients.

        Will be called in `self.__init__()` if cache file does not exist or
            refresh_cache is True.

        This function will first call `self.parse_basic_info()` to parse the
        basic patient information, and then call `self.parse_[table_name]()` to
        parse the table with name `table_name`. Both `self.parse_basic_info()` and
        `self.parse_[table_name]()` should be implemented in the subclass.

        Returns:
           A dict mapping patient_id to `Patient` object.
        """
        pandarallel.initialize(progress_bar=False)

        # patients is a dict of Patient objects indexed by patient_id
        patients: Dict[str, Patient] = dict()
        # process basic information (e.g., patients and visits)
        tic = time.time()
        patients = self.parse_basic_info(patients)
        print(
            "finish basic patient information parsing : {}s".format(time.time() - tic)
        )

        # process clinical tables
        for table in self.tables:
            try:
                # use lower case for function name
                tic = time.time()
                patients = getattr(self, f"parse_{table.lower()}")(patients)
                print(f"finish parsing {table} : {time.time() - tic}s")
            except AttributeError:
                raise NotImplementedError(
                    f"Parser for table {table} is not implemented yet."
                )
        return patients

    def _add_events_to_patient_dict(
        self,
        patient_dict: Dict[str, Patient],
        group_df: pd.DataFrame,
    ) -> Dict[str, Patient]:
        """Helper function which adds the events column of a df.groupby object to the patient dict.

        Will be called at the end of each `self.parse_[table_name]()` function.

        Args:
            patient_dict: a dict mapping patient_id to `Patient` object.
            group_df: a df.groupby object, having two columns: patient_id and events.
                - the patient_id column is the index of the patient
                - the events column is a list of <Event> objects

        Returns:
            The updated patient dict.
        """
        for _, events in group_df.items():
            for event in events:
                patient_dict = self._add_event_to_patient_dict(patient_dict, event)
        
        return patient_dict

    @staticmethod
    def _add_event_to_patient_dict(
        patient_dict: Dict[str, Patient],
        event: Event,
    ) -> Dict[str, Patient]:
        """Helper function which adds an event to the patient dict.

        Will be called in `self._add_events_to_patient_dict`.

        Note that if the patient of the event is not in the patient dict, or the
        visit of the event is not in the patient, this function will do nothing.

        Args:
            patient_dict: a dict mapping patient_id to `Patient` object.
            event: an event to be added to the patient dict.

        Returns:
            The updated patient dict.
        """
        patient_id = event.patient_id
        try:
            patient_dict[patient_id].add_event(event)
        except KeyError:
            pass
        return patient_dict

    def _convert_code_in_patient_dict(
        self,
        patients: Dict[str, Patient],
    ) -> Dict[str, Patient]:
        """Helper function which converts the codes for all patients.

        The codes to be converted are specified in `self.code_mapping`.

        Will be called in `self.__init__()` after `self.parse_tables()`.

        Args:
            patients: a dict mapping patient_id to `Patient` object.

        Returns:
            The updated patient dict.
        """
        for p_id, patient in tqdm(patients.items(), desc="Mapping codes"):
            patients[p_id] = self._convert_code_in_patient(patient)
        return patients

    def _convert_code_in_patient(self, patient: Patient) -> Patient:
        """Helper function which converts the codes for a single patient.

        Will be called in `self._convert_code_in_patient_dict()`.

        Args:
            patient:a `Patient` object.

        Returns:
            The updated `Patient` object. 
        """
        for visit in patient:
            for table in visit.available_tables:
                all_mapped_events = []
                for event in visit.get_event_list(table):
                    # an event may be mapped to multiple events after code conversion
                    mapped_events: List[Event]
                    mapped_events = self._convert_code_in_event(event)
                    all_mapped_events.extend(mapped_events)
                visit.set_event_list(table, all_mapped_events)
        return patient

    def _convert_code_in_event(self, event: Event) -> List[Event]:
        """Helper function which converts the code for a single event.

        Note that an event may be mapped to multiple events after code conversion.

        Will be called in `self._convert_code_in_patient()`.

        Args:
            event: an `Event` object.

        Returns:
            A list of `Event` objects after code conversion.
        """
        src_vocab = event.vocabulary
        if src_vocab in self.code_mapping:
            target = self.code_mapping[src_vocab]
            if isinstance(target, tuple):
                tgt_vocab, kwargs = target
                source_kwargs = kwargs.get("source_kwargs", {})
                target_kwargs = kwargs.get("target_kwargs", {})
            else:
                tgt_vocab = self.code_mapping[src_vocab]
                source_kwargs = {}
                target_kwargs = {}
            code_mapping_tool = self.code_mapping_tools[f"{src_vocab}_{tgt_vocab}"]
            mapped_code_list = code_mapping_tool.map(
                event.code, source_kwargs=source_kwargs, target_kwargs=target_kwargs
            )
            mapped_event_list = [deepcopy(event) for _ in range(len(mapped_code_list))]
            for i, mapped_event in enumerate(mapped_event_list):
                mapped_event.code = mapped_code_list[i]
                mapped_event.vocabulary = tgt_vocab
            
            # update the code vocs
            for key, value in self.code_vocs.items():
                if value == src_vocab:
                    self.code_vocs[key] = tgt_vocab

            return mapped_event_list
        # TODO: should normalize the code here
        return [event]

    def _load_data_from_db(self, query):
        """Helper function to load data from db using SQLAlchemy."""
        with self.engine.connect() as connection:
            return pd.read_sql_query(query, connection)

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses person, visit_occurrence, and death tables.

        Will be called in `self.parse_tables()`

        Docs:
            - person: http://ohdsi.github.io/CommonDataModel/cdm53.html#PERSON
            - visit_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#VISIT_OCCURRENCE
            - death: http://ohdsi.github.io/CommonDataModel/cdm53.html#DEATH

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict. 
        """

        # read person table
        person_df = pd.read_csv(
            os.path.join(self.root, "person.csv" + ('.gz' if self.use_compressed else '')),
            dtype={"person_id": str},
            nrows=1000 if self.dev else None,
            sep=self.sep,
        )
        # read visit_occurrence table
        visit_occurrence_df = pd.read_csv(
            os.path.join(self.root, "visit_occurrence.csv" + ('.gz' if self.use_compressed else '')),
            dtype={"person_id": str, "visit_occurrence_id": str},
            sep=self.sep,
        )
        # read death table
        death_df = pd.read_csv(
            os.path.join(self.root, "death.csv" + ('.gz' if self.use_compressed else '')),
            sep=self.sep,
            dtype={"person_id": str},
        )
        # merge
        df = pd.merge(person_df, visit_occurrence_df, on="person_id", how="left")
        df = pd.merge(df, death_df, on="person_id", how="left")
        # sort by admission time
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "visit_start_datetime"], ascending=True
        )

        # query = """
        # SELECT p.person_id, p.year_of_birth, p.month_of_birth, p.day_of_birth,
        #     p.gender_concept_id, p.race_concept_id, v.visit_occurrence_id,
        #     v.visit_start_datetime, v.visit_end_datetime, d.death_date
        # FROM omop.person p
        # LEFT JOIN omop.visit_occurrence v ON p.person_id = v.person_id
        # LEFT JOIN omop.death d ON p.person_id = d.person_id
        # ORDER BY p.person_id, v.visit_occurrence_id, v.visit_start_datetime;
        # """
        # df = self._load_data_from_db(query)
        # print(df)

        # group by patient
        df_group = df.groupby("person_id")

        # parallel unit of basic informatin (per patient)
        def basic_unit(p_info):
            p_id = p_info["person_id"].values[0]
            birth_y = p_info["year_of_birth"].values[0]
            birth_m = p_info["month_of_birth"].values[0]
            birth_d = p_info["day_of_birth"].values[0]
            birth_date = f"{birth_y}-{birth_m}-{birth_d}"
            patient = Patient(
                patient_id=p_id,
                # no exact time, use 00:00:00
                birth_datetime=strptime(birth_date),
                death_datetime=strptime(p_info["death_date"].values[0]),
                gender=p_info["gender_concept_id"].values[0],
                ethnicity=p_info["race_concept_id"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                death_date = v_info["death_date"].values[0]
                visit_start_date = v_info["visit_start_date"].values[0]
                visit_end_date = v_info["visit_end_date"].values[0]
                if pd.isna(death_date):
                    discharge_status = 0
                elif death_date > visit_end_date:
                    discharge_status = 0
                else:
                    discharge_status = 1
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(visit_start_date),
                    discharge_time=strptime(visit_end_date),
                    discharge_status=discharge_status,
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(lambda x: basic_unit(x))

        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_condition_occurrence(
        self, patients: Dict[str, Patient]
    ) -> Dict[str, Patient]:
        """Helper function which parses condition_occurrence table.

        Will be called in `self.parse_tables()`

        Docs:
            - condition_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#CONDITION_OCCURRENCE

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "condition_occurrence"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv" + ('.gz' if self.use_compressed else '')),
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "condition_concept_id": str,
            },
            sep=self.sep,
        )
        # drop rows with missing values 
        df = df.dropna(
            subset=["person_id", "visit_occurrence_id", "condition_concept_id"]
        )
        # sort by condition_start_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "condition_start_datetime"],
            ascending=True,
        )
        # group by patient and visit
        group_df = df.groupby("person_id")

        # parallel unit of condition occurrence (per patient)
        def condition_unit(p_info):
            p_id = p_info["person_id"].values[0]

            events = []
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                for timestamp, code in zip(
                    v_info["condition_start_datetime"], v_info["condition_concept_id"]
                ):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="CONDITION_CONCEPT_ID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: condition_unit(x))
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_procedure_occurrence(
        self, patients: Dict[str, Patient]
    ) -> Dict[str, Patient]:
        """Helper function which parses procedure_occurrence table.

        Will be called in `self.parse_tables()`

        Docs:
            - procedure_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#PROCEDURE_OCCURRENCE

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "procedure_occurrence"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv" + ('.gz' if self.use_compressed else '')),
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "procedure_concept_id": str,
            },
            sep=self.sep,
        )
        # drop rows with missing values
        df = df.dropna(
            subset=["person_id", "visit_occurrence_id", "procedure_concept_id"]
        )
        # sort by procedure_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "procedure_datetime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("person_id")

        # parallel unit of procedure occurrence (per patient)
        def procedure_unit(p_info):
            p_id = p_info["person_id"].values[0]

            events = []
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                for timestamp, code in zip(
                    v_info["procedure_datetime"], v_info["procedure_concept_id"]
                ):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="PROCEDURE_CONCEPT_ID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: procedure_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_drug_exposure(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses drug_exposure table.

        Will be called in `self.parse_tables()`

        Docs:
            - procedure_occurrence: http://ohdsi.github.io/CommonDataModel/cdm53.html#DRUG_EXPOSURE

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "drug_exposure"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv" + ('.gz' if self.use_compressed else '')),
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "drug_concept_id": str,
            },
            sep=self.sep,
        )
        # drop rows with missing values
        df = df.dropna(subset=["person_id", "visit_occurrence_id", "drug_concept_id"])
        # sort by drug_exposure_start_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "drug_exposure_start_datetime"],
            ascending=True,
        )
        # group by patient and visit
        group_df = df.groupby("person_id")

        # parallel unit of drug exposure (per patient)
        def drug_unit(p_info):
            p_id = p_info["person_id"].values[0]

            events = []
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                for timestamp, code in zip(
                    v_info["drug_exposure_start_datetime"], v_info["drug_concept_id"]
                ):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="DRUG_CONCEPT_ID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: drug_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_measurement(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses measurement table.

        Will be called in `self.parse_tables()`

        Docs:
            - measurement: http://ohdsi.github.io/CommonDataModel/cdm53.html#MEASUREMENT

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "measurement"

        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv" + ('.gz' if self.use_compressed else '')),
            dtype={
                "person_id": str,
                "visit_occurrence_id": str,
                "measurement_concept_id": str,
            },
            sep=self.sep)

        # drop rows with missing values
        df = df.dropna(
            subset=["person_id", "visit_occurrence_id", "measurement_concept_id"]
        )
        # sort by measurement_datetime
        df = df.sort_values(
            ["person_id", "visit_occurrence_id", "measurement_datetime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("person_id")

        # parallel unit of measurement (per patient)
        def measurement_unit(p_info):
            p_id = p_info["person_id"].values[0]

            events = []
            for v_id, v_info in p_info.groupby("visit_occurrence_id"):
                for timestamp, code in zip(
                    v_info["measurement_datetime"], v_info["measurement_concept_id"]
                ):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MEASUREMENT_CONCEPT_ID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: measurement_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    @property
    def available_tables(self) -> List[str]:
        """Returns a list of available tables for the dataset.

        Returns:
            List of available tables.
        """
        tables = []
        for patient in self.patients.values():
            tables.extend(patient.available_tables)
        return list(set(tables))

    def __str__(self):
        """Prints some information of the dataset."""
        return f"Base dataset {self.dataset_name}"

    def stat(self) -> str:
        """Returns some statistics of the base dataset."""
        lines = list()
        lines.append("")
        lines.append(f"Statistics of base dataset (dev={self.dev}):")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Number of patients: {len(self.patients)}")
        num_visits = [len(p) for p in self.patients.values()]
        lines.append(f"\t- Number of visits: {sum(num_visits)}")
        lines.append(
            f"\t- Number of visits per patient: {sum(num_visits) / len(num_visits):.4f}"
        )
        for table in self.tables:
            num_events = [
                len(v.get_event_list(table)) for p in self.patients.values() for v in p
            ]
            lines.append(
                f"\t- Number of events per visit in {table}: "
                f"{sum(num_events) / len(num_events):.4f}"
            )
        lines.append("")
        print("\n".join(lines))
        return "\n".join(lines)

    @staticmethod
    def info():
        """Prints the output format."""
        print(INFO_MSG)

    def set_task(
        self,
        task_fn: Callable,
        task_name: Optional[str] = None,
    ) -> SampleEHRDataset:
        """Processes the base dataset to generate the task-specific sample dataset.

        This function should be called by the user after the base dataset is
        initialized. It will iterate through all patients in the base dataset
        and call `task_fn` which should be implemented by the specific task.

        Args:
            task_fn: a function that takes a single patient and returns a
                list of samples (each sample is a dict with patient_id, visit_id,
                and other task-specific attributes as key). The samples will be
                concatenated to form the sample dataset.
            task_name: the name of the task. If None, the name of the task
                function will be used.
 
        Returns:
            sample_dataset: the task-specific sample dataset.

        Note:
            In `task_fn`, a patient may be converted to multiple samples, e.g.,
                a patient with three visits may be converted to three samples
                ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]).
                Patients can also be excluded from the task dataset by returning
                an empty list.
        """
        if task_name is None:
            task_name = task_fn.__name__
        samples = []
        for patient_id, patient in tqdm(
            self.patients.items(), desc=f"Generating samples for {task_name}"
        ):
            samples.extend(task_fn(patient))

        sample_dataset = SampleEHRDataset(
            samples=samples,
            code_vocs=self.code_vocs,
            dataset_name=self.dataset_name,
            task_name=task_name,
        )
        return sample_dataset
