import torch

class CubeFace:
    U = 0
    R = 1
    F = 2
    D = 3
    L = 4
    B = 5

    COUNT = 6

class CubeSet:
    @torch.no_grad()
    def __init__(self, n, device):
        self.n = n
        self.device = device

        # Position permutations for each corner and edge
        self.corner_perm = torch.arange(0, 8, device=device, dtype=torch.uint8).unsqueeze(0).repeat(self.n, 1)
        self.edge_perm = torch.arange(0, 12, device=device, dtype=torch.uint8).unsqueeze(0).repeat(self.n, 1)

        # Range: [0, 2]
        # The number of clockwise twists away from the white or yellow face is being correctly oriented
        self.corner_rot = torch.zeros(self.n, 8, device=device, dtype=torch.uint8)

        # Range: [0, 1]
        # Equal to 0 if oriented correctly, 1 if not
        self.edge_rot = torch.zeros(self.n, 12, device=device, dtype=torch.uint8)

        #####################
        # Big constant tensors generated from the rust codebase
        # NOTE: Additionally, there's 1 extra dummy move on the end if you want the turn to do nothing

        self._turn_corner_perm = torch.tensor(
            [
                [[1, 2, 3, 0, 4, 5, 6, 7], [3, 0, 1, 2, 4, 5, 6, 7], [2, 3, 0, 1, 4, 5, 6, 7]],
                [[3, 1, 2, 7, 0, 5, 6, 4], [4, 1, 2, 0, 7, 5, 6, 3], [7, 1, 2, 4, 3, 5, 6, 0]],
                [[4, 0, 2, 3, 5, 1, 6, 7], [1, 5, 2, 3, 0, 4, 6, 7], [5, 4, 2, 3, 1, 0, 6, 7]],
                [[0, 1, 2, 3, 7, 4, 5, 6], [0, 1, 2, 3, 5, 6, 7, 4], [0, 1, 2, 3, 6, 7, 4, 5]],
                [[0, 5, 1, 3, 4, 6, 2, 7], [0, 2, 6, 3, 4, 1, 5, 7], [0, 6, 5, 3, 4, 2, 1, 7]],
                [[0, 1, 6, 2, 4, 5, 7, 3], [0, 1, 3, 7, 4, 5, 2, 6], [0, 1, 7, 6, 4, 5, 3, 2]],

                # Dummy
                [list(range(8))]*3
            ], device=device, dtype=torch.long
        )

        self._turn_edge_perm = torch.tensor(
            [
                [[1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11], [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
                 [2, 3, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11]],
                [[11, 1, 2, 3, 8, 5, 6, 7, 0, 9, 10, 4], [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
                 [4, 1, 2, 3, 0, 5, 6, 7, 11, 9, 10, 8]],
                [[0, 8, 2, 3, 4, 9, 6, 7, 5, 1, 10, 11], [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
                 [0, 5, 2, 3, 4, 1, 6, 7, 9, 8, 10, 11]],
                [[0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11], [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
                 [0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11]],
                [[0, 1, 9, 3, 4, 5, 10, 7, 8, 6, 2, 11], [0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11],
                 [0, 1, 6, 3, 4, 5, 2, 7, 8, 10, 9, 11]],
                [[0, 1, 2, 10, 4, 5, 6, 11, 8, 9, 7, 3], [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
                 [0, 1, 2, 7, 4, 5, 6, 3, 8, 9, 11, 10]],

                # Dummy
                [list(range(12))]*3
            ], device=device, dtype=torch.long
        )

        self._turn_corner_rot = torch.tensor(
            [
                [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
                [[2, 0, 0, 1, 1, 0, 0, 2], [2, 0, 0, 1, 1, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0]],
                [[1, 2, 0, 0, 2, 1, 0, 0], [1, 2, 0, 0, 2, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 1, 2, 0, 0, 2, 1, 0], [0, 1, 2, 0, 0, 2, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 1, 2, 0, 0, 2, 1], [0, 0, 1, 2, 0, 0, 2, 1], [0, 0, 0, 0, 0, 0, 0, 0]],

                # Dummy
                [[0] * 8] * 3
            ], device=device, dtype=torch.uint8
        )

        self._turn_edge_rot = torch.tensor(
            [
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

                # Dummy
                [[0] * 12] * 3
            ], device=device, dtype=torch.uint8
        )

        self._corner_rot_shift_map = torch.tensor(
            [ # TODO: This is literally just shifting arange(3) lol, this does not need to be stored anywhere
                [0, 1, 2], [1, 2, 0], [2, 0, 1],
            ], device=device, dtype=torch.long
        )

    @torch.no_grad()
    # Do 1 turn for all cubes
    def do_turn(self, move_indices: torch.Tensor):
        if isinstance(move_indices, int):
            # Turn a single integer into a set of face indices for all cubes
            face_idx = move_indices
            move_indices = torch.zeros((self.n,), device=self.device, dtype=torch.long)
            move_indices += face_idx * 3
        assert move_indices.shape == (self.n,)

        move_indices_tup = (move_indices // 3, move_indices % 3)
        corner_perm_maps = self._turn_corner_perm[move_indices_tup]
        edge_perm_maps = self._turn_edge_perm[move_indices_tup]
        corner_rot_adds = self._turn_corner_rot[move_indices_tup]
        edge_rot_adds = self._turn_edge_rot[move_indices_tup]

        if True:
            # Fully-vectorized version

            # Corners
            factor_corner_idc = self.corner_perm.to(torch.long)
            perm_corn_rot = torch.gather(corner_rot_adds, 1, factor_corner_idc)
            self.corner_rot = self._corner_rot_shift_map[
                perm_corn_rot.to(torch.long),
                self.corner_rot.to(torch.long)
            ]
            self.corner_perm = torch.gather(corner_perm_maps, 1, factor_corner_idc)

            # Edges
            factor_edge_idc = self.edge_perm.to(torch.long)
            edge_rot_add_vals = torch.gather(edge_rot_adds, 1, factor_edge_idc)
            self.edge_rot = (self.edge_rot + edge_rot_add_vals) % 2
            self.edge_perm = torch.gather(edge_perm_maps, 1, factor_edge_idc)
        elif True:
            # Half-vectorized version
            # Corners
            for i in range(self.n):
                factor_corner_idc = self.corner_perm[i].to(torch.long)
                perm_corn_rot = corner_rot_adds[i][factor_corner_idc]
                shifts = self._corner_rot_shift_map[perm_corn_rot.to(torch.long)]

                self.corner_rot[i] = shifts[torch.arange(8), self.corner_rot[i].to(torch.long)]

                # Permute corners
                self.corner_perm[i] = corner_perm_maps[i][factor_corner_idc]

            # Edges
            for i in range(self.n):
                factor_edge_idc = self.edge_perm[i].to(torch.long)
                self.edge_rot[i] = (self.edge_rot[i] + edge_rot_adds[i][factor_edge_idc]) % 2
                self.edge_perm[i] = edge_perm_maps[i][factor_edge_idc]
        else:
            # Slow reference

            # Corners
            for i in range(self.n):
                for j in range(8):
                    factor_corner_idx = self.corner_perm[i, j].to(torch.long)

                    # Rotate corner
                    perm_corn_rot = corner_rot_adds[i][factor_corner_idx]
                    shifts = self._corner_rot_shift_map[perm_corn_rot.to(torch.long)]
                    self.corner_rot[i, j] = shifts[self.corner_rot[i, j].to(torch.long)]

                    # Permute corner
                    self.corner_perm[i, j] = corner_perm_maps[i][factor_corner_idx]

            # Edges
            for i in range(self.n):
                for j in range(12):
                    factor_edge_idx = self.edge_perm[i, j].to(torch.long)
                    self.edge_rot[i, j] = (self.edge_rot[i, j] + edge_rot_adds[i][factor_edge_idx]) % 2

                    # Permute edge
                    self.edge_perm[i, j] = edge_perm_maps[i][factor_edge_idx]

    def do_turns(self, move_indices_set: torch.Tensor):
        num_moves = move_indices_set.shape[1]
        assert list(move_indices_set.shape) == [self.n, num_moves]
        for i in range(num_moves):
            self.do_turn(move_indices_set[:, i])

    def get_vals_dict(self, cube_idx) -> dict[str, list]:
        return {
            "corner_perm": self.corner_perm[cube_idx].tolist(),
            "edge_perm": self.edge_perm[cube_idx].tolist(),
            "corner_rot": self.corner_rot[cube_idx].tolist(),
            "edge_rot": self.edge_rot[cube_idx].tolist(),
        }

    NUM_TOKEN_TYPES = 1 + (8*3) + (12*2)

    @torch.no_grad()
    def get_obs(self):
        corner_tokens = self.corner_perm.to(torch.long) * 3 + self.corner_rot.to(torch.long)
        edge_tokens = self.edge_perm.to(torch.long) * 2 + self.edge_rot.to(torch.long)

        result = torch.concat([
            1 + corner_tokens,
            1 + (8 * 3) + edge_tokens
        ], dim=-1
        )

        assert (result < CubeSet.NUM_TOKEN_TYPES).all()
        return result

    @torch.no_grad()
    def make_scrambling_moves(self, num_moves: torch.Tensor) -> torch.Tensor:
        assert num_moves.shape == (self.n,)
        assert num_moves.dtype == torch.long

        max_moves = num_moves.max()
        random_turn_faces = torch.randint(0, 6, size=(self.n, max_moves), device=self.device)

        # We need to make sure we don't generate two turns on a consecutive face
        # So, we go to 5 and add one if we reach the previous face
        # This ensures no repeated faces without trial and error
        for move_idx in range(1, max_moves):
            prev_faces = random_turn_faces[:, move_idx - 1]
            new_faces = torch.randint(0, 5, size=(self.n,), device=self.device)

            # If new_face >= prev_face, increment by 1 to skip the previous face
            new_faces = torch.where(new_faces >= prev_faces, new_faces + 1, new_faces)

            random_turn_faces[:, move_idx] = new_faces

        if 1: # Mask out moves that go beyond the limit for that column
            # TODO: There's probably a smarter, simpler way to do this
            random_turn_ranges = torch.arange(start=1, end=(max_moves+1), device=self.device).unsqueeze(0).repeat(self.n, 1)
            random_turn_mask = random_turn_ranges <= num_moves.unsqueeze(-1)

            random_turn_faces = (random_turn_faces * random_turn_mask) + (6 * random_turn_mask.logical_not())

        random_turn_dirs = torch.randint(0, 3, size=(self.n, max_moves), device=self.device)
        random_turns = (random_turn_faces * 3) + random_turn_dirs
        return random_turns

    # Returns scrambling moves
    @torch.no_grad()
    def scramble_all(self, num_moves: torch.Tensor) -> torch.Tensor:
        random_turns = self.make_scrambling_moves(num_moves)
        self.do_turns(random_turns)
        return random_turns

    @torch.no_grad()
    def get_solved_mask(self):
        target_corners = torch.arange(8, device=self.device, dtype=torch.uint8)
        target_edges = torch.arange(12, device=self.device, dtype=torch.uint8)

        solved_corner_rot = (self.corner_rot == 0).all(dim=-1)
        solved_edge_rot = (self.edge_rot == 0).all(dim=-1)
        solved_corner_perm = (self.corner_perm == target_corners).all(dim=-1)
        solved_edge_perm = (self.edge_perm == target_edges).all(dim=-1)

        return solved_corner_rot & solved_edge_rot & solved_corner_perm & solved_edge_perm

    @torch.no_grad()
    def get_hashes_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs
        primes = torch.tensor([ # From https://bigprimes.org/
            949728150821393, 506950044530453, 392968686723523, 991925574461279, 711090494984593, 935744767383787,
            103991894025197, 633626152995593, 665639499714583, 766753476352667, 323689212341219, 606266698827403,
            479563265278901, 732690361504733, 127947114014309, 709219295366119, 734423030423281, 386303233229141,
            356897023341371, 265423778035543
        ], device=self.device, dtype=torch.int64)
        hashes = (obs * primes).sum(dim=-1)
        return hashes

class CubeSet3D:
    @torch.no_grad()
    def __init__(self, n, device):
        self.n = n
        self.device = device

        self._edge_perm_rots = torch.tensor([
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
            [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],

            [[0, 1, 0], [-1, 0, 0], [0, 0, -1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        ], device=device, dtype=torch.float32)

        rot_template = torch.eye(3, device=device).reshape(1, 1, 3, 3)
        self.corner_rots = rot_template.repeat(n, 8, 1, 1)
        self.edge_rots = self._edge_perm_rots.reshape(1, 1, 3, 3).repeat(n, 12, 1, 1)

        self.edge_rots = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
        ], device=device, dtype=torch.float32)

        self._turn_rots = torch.tensor([
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # U
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],  # R
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],  # F
            [[0, 1, 0], [-1, 0, 0], [0, 0, -1]], # D
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]],  # L
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],  # B
        ], device=device, dtype=torch.float32)

        self._turn_corner_mask = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1],
        ], device=device, dtype=torch.bool)

        self._turn_edge_mask = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        ], device=device, dtype=torch.bool)

    @torch.no_grad()
    def do_clockwise_turn(self, face_indices: torch.Tensor):
        assert face_indices.shape == (self.n,)
        assert face_indices.dtype == torch.long

        turn_rots = self._turn_rots[face_indices]
        turn_corner_masks = self._turn_corner_mask[face_indices]
        turn_edge_masks = self._turn_edge_mask[face_indices]

        if True:
            # Fully-vectorized version

            # Rotate everything
            turn_rots = turn_rots.view(self.n, 1, 3, 3)
            new_corner_rots = torch.matmul(self.corner_rots, turn_rots)
            new_edge_rots = torch.matmul(self.edge_rots, turn_rots)

            # Mask the actual effect when applying to the turn's respective layer of edges/corners
            turn_corner_masks = turn_corner_masks.view(self.n, 8, 1, 1)
            turn_edge_masks = turn_edge_masks.view(self.n, 12, 1, 1)
            self.corner_rots = (self.corner_rots * turn_corner_masks.logical_not()) + (new_corner_rots * turn_corner_masks)
            self.edge_rots = (self.edge_rots * turn_edge_masks.logical_not()) + (new_edge_rots * turn_edge_masks)
        else:
            # Reference version

            for i in range(self.n):
                # Rotate everything
                turn_rot = turn_rots[i].view(1, 3, 3)
                new_corner_rots = torch.matmul(self.corner_rots[i], turn_rot)
                new_edge_rots = torch.matmul(self.edge_rots[i], turn_rot)


                # Mask the actual effect when applying to the turn's respective layer of edges/corners
                turn_corner_mask = turn_corner_masks[i].view(8, 1, 1)
                turn_edge_mask = turn_edge_masks[i].view(12, 1, 1)
                self.corner_rots[i] = (self.corner_rots[i] * turn_corner_mask.logical_not()) + (new_corner_rots * turn_corner_mask)
                self.edge_rots[i] = (self.edge_rots[i] * turn_edge_mask.logical_not()) + (new_edge_rots * turn_edge_mask)

if __name__ == "__main__":
    DEVICE = "cuda"

    n = 2
    num_moves = 1
    cube = CubeSet(n, DEVICE)
    cube.scramble_all(
        torch.tensor([
            5, 10
        ], device=DEVICE)
    )

    print("Solve mask:", cube.get_solved_mask().tolist())
    print("Vals dict:", cube.get_vals_dict(0))