    def get_training_batch(self, online_filtering: bool = True, max_times_to_make_a_batch=8) -> DataProto:
        self.async_rollout_manager.outside_wakeup()

        timing_raw = {}

        filtered_out_batch = None
        final_batch = None
        selected_gen_batch = None
        filtered_out_gen_batch = None

        too_bad_sample_count, too_good_sample_count, total_num_samples = 0, 0, 0

        try_to_make_a_batch_times = 0
        if len(self.num_rollout_samples_before_filtering_history) == 0:
            if online_filtering:
                num_estimate_total_rollout_samples = 1 * self.config.data.train_batch_size
            else:
                num_estimate_total_rollout_samples = self.config.data.train_batch_size
        else:
            num_estimate_total_rollout_samples = np.mean(self.num_rollout_samples_before_filtering_history)
        cur_batch: Optional[DataProto] = self.pending_samples
        self.pending_samples = None

        num_rollout_samples = 0 if cur_batch is None else len(cur_batch)

        print(
            f"Start fetching a batch - prev pending samples {num_rollout_samples}, "
            f"estimate total rollout samples {num_estimate_total_rollout_samples:.2f}"
        )
        while try_to_make_a_batch_times < max_times_to_make_a_batch:
            try_to_make_a_batch_times += 1

            if online_filtering:
                print(
                    f"trial {try_to_make_a_batch_times}/{max_times_to_make_a_batch} - "
                    f"estimate new: {num_estimate_total_rollout_samples}, "
                    f"current fetched samples: {num_rollout_samples}"
                )

            while num_rollout_samples < num_estimate_total_rollout_samples:
                try:
                    _batch_dict = next(self.data_iterator)
                except StopIteration:
                    self.num_dataloader_exhausted += 1
                    print(f"train_dataloader StopIteration times: {self.num_dataloader_exhausted}")
                    self.data_iterator = iter(self.train_dataloader)
                    _batch_dict = next(self.data_iterator)

                _batch: DataProto = DataProto.from_single_dict(_batch_dict)
                num_rollout_samples += len(_batch)
                if cur_batch is None:
                    cur_batch = _batch
                else:
                    cur_batch = DataProto.concat([cur_batch, _batch])

            num_estimate_total_rollout_samples += self.config.data.train_batch_size

            batch = cur_batch
            assert len(batch) == num_rollout_samples

            # batch: DataProto = DataProto.from_single_dict(batch_dict)

            # add uid to batch
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

            gen_batch = self._get_gen_batch(batch)

            # pass global_steps to trace
            gen_batch.meta_info["global_steps"] = self.global_steps

            # the states need deepcopy, and we do it inside the AgentLoopWorker
            gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

            # is_last_step = self.global_steps >= self.total_training_steps
            with marked_timer("gen", timing_raw, color="red"):
                assert self.async_rollout_mode is True
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)

            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch = batch.union(gen_batch_output)

            # for state in batch.non_tensor_batch["state"]:
            #     state.environment.screenshot.close()

            if not online_filtering:
                self.async_rollout_manager.outside_sleep()
                return batch, gen_batch, {}

            # filtering logic start -- todo implement in a function
            key_for_filtering = "success"
            assert key_for_filtering in gen_batch_output.meta_info["reward_extra_keys"]
            scores_for_filtering = gen_batch_output.non_tensor_batch[key_for_filtering]
            num_trajs = self.config.actor_rollout_ref.rollout.n
            scores_for_filtering = np.array(scores_for_filtering).reshape(-1, num_trajs)  # (batch_size, n)
            keep_mask = np.logical_and(
                np.any(scores_for_filtering == 1.0, axis=1), np.any(scores_for_filtering == 0.0, axis=1)
            )
            too_good_sample_count += np.sum(np.all(scores_for_filtering == 1.0, axis=1))  # all 1.0
            too_bad_sample_count += np.sum(np.all(scores_for_filtering == 0.0, axis=1))
            total_num_samples += len(keep_mask)
            # filtering logic end
            print(
                f"trial {try_to_make_a_batch_times}/{max_times_to_make_a_batch} - "
                f"filter out {np.sum(~keep_mask)} samples from {len(keep_mask)} samples"
            )

            # according to keep_mask, only keep at most batch_size samples, others will be put into pending_samples
            cur_selected_num_samples = 0 if final_batch is None else len(final_batch) // num_trajs
            if np.sum(keep_mask) + cur_selected_num_samples > self.config.data.train_batch_size:
                need_num_samples = self.config.data.train_batch_size - cur_selected_num_samples
                cumsum = np.cumsum(keep_mask)
                split_idx = np.searchsorted(cumsum, need_num_samples, side="right")
                keep_mask[split_idx:] = False
                assert sum(keep_mask) == need_num_samples
                self.pending_samples = cur_batch[split_idx:]

            # expand keep_mask to num_trajs
            keep_mask_expanded = np.repeat(keep_mask, num_trajs)

            if np.sum(~keep_mask) > 0:
                if filtered_out_batch is None:
                    filtered_out_batch = batch.select_idxs(~keep_mask_expanded)
                    filtered_out_gen_batch = gen_batch.select_idxs(~keep_mask)
                else:
                    filtered_out_batch = DataProto.concat([filtered_out_batch, batch.select_idxs(~keep_mask_expanded)])
                    filtered_out_gen_batch = DataProto.concat(
                        [filtered_out_gen_batch, gen_batch.select_idxs(~keep_mask)]
                    )
            # add kept samples to final_batch
            if np.sum(keep_mask) > 0:
                if final_batch is None:
                    final_batch = batch.select_idxs(keep_mask_expanded)
                    selected_gen_batch = gen_batch.select_idxs(keep_mask)
                else:
                    final_batch = DataProto.concat([final_batch, batch.select_idxs(keep_mask_expanded)])
                    selected_gen_batch = DataProto.concat([selected_gen_batch, gen_batch.select_idxs(keep_mask)])

            cur_selected_num_samples = 0 if final_batch is None else len(final_batch) // num_trajs
            cur_filtered_num_samples = 0 if filtered_out_batch is None else len(filtered_out_batch) // num_trajs
            print(
                f"trial {try_to_make_a_batch_times}: cur_batch={cur_selected_num_samples}, total filtered={cur_filtered_num_samples}"
            )

            del batch, gen_batch, gen_batch_output

            if cur_selected_num_samples >= self.config.data.train_batch_size:
                break

            # we don't need to keep all filtered samples if we already have enough samples
            if (
                filtered_out_batch is not None
                and (cur_selected_num_samples + cur_filtered_num_samples) >= self.config.data.train_batch_size
            ):
                # truncate filtered_out_batch to save memory
                need_num_samples = self.config.data.train_batch_size - cur_selected_num_samples
                filtered_out_batch = filtered_out_batch.select_idxs(np.arange(need_num_samples * num_trajs))
                filtered_out_gen_batch = filtered_out_gen_batch.select_idxs(np.arange(need_num_samples))

        self.num_rollout_samples_before_filtering_history.append(total_num_samples)
        self.num_rollout_samples_before_filtering_history = self.num_rollout_samples_before_filtering_history[-3:]

        batch_num_samples = 0 if final_batch is None else len(final_batch) // num_trajs
        filtered_num_samples = 0 if filtered_out_batch is None else len(filtered_out_batch) // num_trajs
        print(f"Got {batch_num_samples} samples after {max_times_to_make_a_batch} trials")

        if batch_num_samples < self.config.data.train_batch_size:
            print(f"less than target size {self.config.data.train_batch_size}, select filtered samples to fill")
            need_num_samples = self.config.data.train_batch_size - batch_num_samples
            assert need_num_samples <= filtered_num_samples
            if final_batch is not None:
                final_batch = DataProto.concat(
                    [final_batch, filtered_out_batch.select_idxs(np.arange(need_num_samples * num_trajs))]
                )
                selected_gen_batch = DataProto.concat(
                    [selected_gen_batch, filtered_out_gen_batch.select_idxs(np.arange(need_num_samples))]
                )
            else:
                final_batch = filtered_out_batch.select_idxs(np.arange(need_num_samples * num_trajs))
                selected_gen_batch = filtered_out_gen_batch.select_idxs(np.arange(need_num_samples))
        elif batch_num_samples > self.config.data.train_batch_size:
            print(f"more than target size {self.config.data.train_batch_size}, truncate to get the final batch")
            final_batch = final_batch.select_idxs(np.arange(self.config.data.train_batch_size * num_trajs))
            selected_gen_batch = selected_gen_batch.select_idxs(np.arange(self.config.data.train_batch_size))
        else:
            print(f"exactly target size {self.config.data.train_batch_size}")

        filtering_info = {
            "rollout/too_bad_sample_ratio": too_bad_sample_count / total_num_samples,
            "rollout/too_good_sample_ratio": too_good_sample_count / total_num_samples,
            "rollout/total_num_samples": total_num_samples,
        }

        if filtered_out_gen_batch is not None:
            # batch.non_tensor_batch.pop("state")
            # imgs_to_close = filtered_out_gen_batch.non_tensor_batch.pop("image")
            # for img in imgs_to_close:
            #     img.close()
            del filtered_out_batch, filtered_out_gen_batch

        self.async_rollout_manager.outside_sleep()
        return final_batch, selected_gen_batch, filtering_info